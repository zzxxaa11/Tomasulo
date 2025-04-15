

import sys
import argparse
from collections import deque

# Configuration and Global Constants


DEFAULT_NF = 4  # Instruction fetch width
DEFAULT_NI = 8 # Instruction queue size
DEFAULT_NW = 4  # Issue width
DEFAULT_NR = 8 # Reorder buffer entries
DEFAULT_NB = 4  # Number of CDB buses

# OpCode enumeration for clarity
OPCODES = {
    'ADD':   1,
    'ADDI':  2,
    'SLT':   3,
    'FADD':  4,
    'FSUB':  5,
    'FMUL':  6,
    'FDIV':  7,
    'FLD':   8,
    'FSD':   9,
    'BNE':   10,
}

# Latency configuration per functional unit (pipeline cycles)
LATENCIES = {
    'INT':    1,  # add, addi, slt
    'LOAD':   1,  # address calc is separate, but actual load is 1 cycle
    'STORE':  1,  # store op
    'FPADD':  3,  # pipelined add/sub
    'FPMUL':  4,  # pipelined multiply
    'FPDIV':  8,  # non-pipelined divide
    'BR':     1,  # branch condition + target
}



class Instruction:

    def __init__(self, raw_text, address):
        self.raw_text = raw_text.strip()
        self.address = address
        # Parsed fields
        self.opcode = None
        self.rd = None
        self.rs1 = None
        self.rs2 = None
        self.imm = None

    def __repr__(self):
        return f"<Instr {self.address:04x}: {self.raw_text}>"


class BranchPredictor2Bit:
    """
    Simple 2-bit saturating counter-based branch predictor with 16-entry BTB.
    The predictor is indexed by bits 7-4 of the instruction address.
    States: 00 -> strongly not taken, 01 -> weakly not taken,
            10 -> weakly taken,     11 -> strongly taken
    """
    def __init__(self):
        self.entries = [2] * 16  # Start with 2 -> 'weakly taken' = 10 in binary
        self.btb = [0] * 16      # Store predicted target addresses

    def _index(self, address):
        # bits 7-4 of address => (address >> 4) & 0xF
        return (address >> 4) & 0xF

    def predict(self, address):
        idx = self._index(address)
        counter = self.entries[idx]
        taken = (counter >= 2)  # 2 or 3 => taken
        target = self.btb[idx]
        return taken, target

    def update(self, address, taken_actual, target_actual):
        idx = self._index(address)
        # Update saturating counter
        if taken_actual:
            if self.entries[idx] < 3:
                self.entries[idx] += 1
        else:
            if self.entries[idx] > 0:
                self.entries[idx] -= 1
        # Update BTB with actual target
        self.btb[idx] = target_actual


class ROBEntry:
    """
    Represents an entry in the Reorder Buffer (ROB).
    """
    def __init__(self, rob_id, instr, dest_reg, dest_prn):
        self.id = rob_id
        self.instr = instr
        self.dest_reg = dest_reg   # Architectural register
        self.dest_prn = dest_prn   # Physical register name
        self.value = None
        self.ready = False
        self.exception = False
        self.done = False

    def __repr__(self):
        return (f"<ROBEntry id={self.id} op={self.instr.opcode} "
                f"destPRN={self.dest_prn} ready={self.ready}>")



class Simulator:
    def __init__(self, nf, ni, nw, nr, nb, program_lines):
        # Config parameters
        self.nf = nf  # fetch width
        self.ni = ni  # instruction queue size
        self.nw = nw  # issue width
        self.nr = nr  # reorder buffer size
        self.nb = nb  # number of CDB lines

        # Program / Memory
        self.instructions = []
        self.data_memory = {}  # map address -> value (assume 32-bit or 64-bit)
        self._parse_program(program_lines)

        # Architectural Registers: 32 integer + 32 floating (?)
        # But project suggests 32 total that can be used either int or float.
        self.arch_reg_count = 32
        # Physical registers: 32 total (p0..p31)
        self.phys_reg_count = 32

        # Rename mapping table: arch_reg -> phys_reg
        self.rename_table = [None]*self.arch_reg_count
        # Start with trivial mapping: R0->p0, R1->p1, ...
        for i in range(self.arch_reg_count):
            self.rename_table[i] = i

        # Free list for physical regs (beyond the initial 32 that might be taken)
        self.free_list = deque(range(self.arch_reg_count, self.phys_reg_count))

        # Actual register file for physical registers
        self.phys_regs = [0]*self.phys_reg_count
        self.phys_reg_valid = [True]*self.phys_reg_count  # Is data valid?

        # Reorder Buffer
        self.ROB = []
        self.rob_head = 0
        self.rob_tail = 0

        # Branch predictor
        self.branch_predictor = BranchPredictor2Bit()

        # Instruction queue
        self.instr_queue = deque()

        # Cycle counter
        self.cycle = 0

        # Stats
        self.stall_res_station = 0
        self.stall_rob_full = 0
        self.cdb_usage_count = 0

        # Reservation Stations placeholders
        # You would create separate station lists for INT, LOAD, STORE, FPADD, FPMUL, FPDIV, BR, etc.
        self.res_stations = {
            'INT':   [],
            'LOAD':  [],
            'STORE': [],
            'FPADD': [],
            'FPMUL': [],
            'FPDIV': [],
            'BR':    [],
        }
        # For demonstration, keep them as empty lists or define station structures.

        # Program Counter
        self.PC = 0x0000

        # State for fetch
        self.fetch_buffer = []

        # Done flag (when program finishes)
        self.done = False

    def _parse_program(self, lines):
        """
        Read lines from the input. Distinguish memory init lines from instructions.
        For example, lines that look like "100, 2" => mem[100] = 2
        vs. "addi R1,R0,24" => an actual instruction
        """
        address_counter = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if it matches mem initialization: e.g. "100, 2"
            if ',' in line and not any(x in line.upper() for x in OPCODES.keys()):
                # It's memory content
                parts = line.split(',')
                if len(parts) == 2:
                    addr_str, val_str = parts
                    try:
                        addr = int(addr_str.strip())
                        val = int(val_str.strip())
                        self.data_memory[addr] = val
                    except ValueError:
                        pass
                continue
            # Otherwise, parse as an instruction
            instr_obj = Instruction(line, address_counter)
            self.instructions.append(instr_obj)
            address_counter += 4  # each instr is 4 bytes (like normal RISC-V)
    
    def reset(self):
        """Reset internal simulator state to rerun or start fresh."""
        self.cycle = 0
        self.PC = 0
        self.ROB.clear()
        self.instr_queue.clear()
        self.fetch_buffer.clear()
        self.done = False
        self.phys_regs = [0]*self.phys_reg_count
        self.phys_reg_valid = [True]*self.phys_reg_count
        # Reset rename table
        for i in range(self.arch_reg_count):
            self.rename_table[i] = i
        self.free_list = deque(range(self.arch_reg_count, self.phys_reg_count))

    def run(self):
        """
        Main loop. At each cycle, run the pipeline stages:
          1) Fetch
          2) Decode/Rename
          3) Issue
          4) Execute
          5) Writeback
          6) Commit
        Stop when the program completes or no active instructions remain.
        """
        while not self.done:
            self.cycle += 1
            self.commit()
            self.writeback()
            self.execute()
            self.issue()
            self.decode()
            self.fetch()

            # Check termination condition
            if self.PC >= 4*len(self.instructions) and not self.ROB:
                self.done = True
        
        print(f"Program finished in {self.cycle + self.stall_res_station + self.stall_rob_full} cycles.")
        print(f"Stalls due to full reservation stations: {self.stall_res_station}")
        print(f"Stalls due to full ROB: {self.stall_rob_full}")
        print(f"CDB usage count: {self.cdb_usage_count}")


    
    def fetch(self):
        """
        Fetch up to self.nf instructions from the instruction list each cycle,
        if available. Put them into a temporary buffer. Next cycle, decode them.
        """
        if len(self.fetch_buffer) < self.nf:
            # Fetch capacity
            to_fetch = self.nf - len(self.fetch_buffer)
            for _ in range(to_fetch):
                if self.PC < 4*len(self.instructions):
                    instr_index = self.PC // 4
                    self.fetch_buffer.append(self.instructions[instr_index])
                    self.PC += 4
                else:
                    break

    def decode(self):
        """
        Move fetched instructions into the instruction queue, performing
        register renaming. Then remove them from fetch_buffer.
        """
        while self.fetch_buffer and len(self.instr_queue) < self.ni:
            instr = self.fetch_buffer.pop(0)
            self.rename_and_enqueue(instr)


    def rename_and_enqueue(self, instr):
        """
        Parse the textual instruction into a structured form,
        perform register renaming, create a ROB entry, and place in the
        instruction queue.
        """
        # Simplified parse
        upper = instr.raw_text.upper()
        tokens = upper.replace(',', ' ').split()

        opcode_str = tokens[0]
        instr.opcode = opcode_str

        if opcode_str in ('ADDI', 'ADD', 'SLT', 'BNE'):
            if len(tokens) >= 4:
                instr.rd = self._reg_index(tokens[1])  
                instr.rs1 = self._reg_index(tokens[2])
                # parse imm or rs2
                try:
                    instr.imm = int(tokens[3])
                except ValueError:
                    # might be label or register
                    instr.rs2 = self._reg_index(tokens[3])
        elif opcode_str in ('FADD', 'FSUB', 'FMUL', 'FDIV'):
            if len(tokens) >= 4:
                instr.rd = self._reg_index(tokens[1])  
                instr.rs1 = self._reg_index(tokens[2])
                instr.rs2 = self._reg_index(tokens[3])
        elif opcode_str in ('FLD', 'FSD'):

            if len(tokens) >= 3:
                instr.rd = self._reg_index(tokens[1])  # F2 -> 2
                mem_part = tokens[2]
 
                off, reg_str = mem_part.split('(')
                reg_str = reg_str.replace(')', '')
                instr.rs1 = self._reg_index(reg_str)  # base
                instr.imm = int(off)  # offset
        else:
            pass

        dest_arch = instr.rd
        new_prn = self.allocate_physical_register()
        if new_prn is None:
            self.stall_rob_full += 1
            return
        # Create ROB entry
        rob_id = len(self.ROB)
        rob_entry = ROBEntry(rob_id, instr, dest_arch, new_prn)
        self.ROB.append(rob_entry)

        # Update rename table for the architectural dest
        if dest_arch is not None and dest_arch != 0:  
            self.rename_table[dest_arch] = new_prn
            self.phys_reg_valid[new_prn] = False

        self.instr_queue.append(instr)

    def issue(self):

        issued_this_cycle = 0
        remaining = len(self.instr_queue)
        for _ in range(remaining):
            if issued_this_cycle >= self.nw:
                break
            instr = self.instr_queue[0]  # peek
            unit_type = self.get_unit_type(instr.opcode)
            if self.has_free_station(unit_type):
                # Remove from IQ
                self.instr_queue.popleft()
                # Reserve a station
                self.reserve_station(unit_type, instr)
                issued_this_cycle += 1
            else:
                # No station available => stall
                self.stall_res_station += 1
                break

    def execute(self):
        for unit_type, stations in self.res_stations.items():
            for station in stations:
                if station['busy']:
                    if not station['operands_ready']:
                        if station.get('src1_ready', False) and station.get('src2_ready', False):
                            station['operands_ready'] = True
                    
                    if station['operands_ready'] and station['remaining_lat'] > 0:
                        # "Executing" => decrement latency
                        station['remaining_lat'] -= 1

                    # If we've just finished
                    if station['operands_ready'] and station['remaining_lat'] == 0:
                        station['completed'] = True
                        station['busy'] = False

    def writeback(self):

        completed_entries = []
        for unit_type, stations in self.res_stations.items():
            for station in stations:
                if station.get('completed', False):
                    completed_entries.append(station)


        if len(completed_entries) > self.nb:
            # slice the list to nb
            completed_entries_to_broadcast = completed_entries[:self.nb]
        else:
            completed_entries_to_broadcast = completed_entries

        for station in completed_entries_to_broadcast:
            instr = station['instr']
            rob_id = station['dest_rob_id']

            result = self.compute_result(instr, station)
            
            rob_entry = self.ROB[rob_id]
            rob_entry.value = result
            rob_entry.ready = True  # Mark it ready to be committed

            self.broadcast_result(rob_id, result)

            # Mark this station as no longer busy
            station['completed'] = False
            station.clear()  # or something that indicates it's free

        self.cdb_usage_count += len(completed_entries_to_broadcast)


    def commit(self):
        """
        Commit instructions from the ROB in order (head-first), if they are ready.
        Free the physical register or handle mispredictions if needed.
        """
        # We can only commit the oldest entry if it's marked done/ready.
        # If branch misprediction is discovered, flush.
        to_commit = []
        for i, rob_entry in enumerate(self.ROB):
            if rob_entry.id != i:
                # not in order -> break
                break
            if rob_entry.ready:
                # can commit
                to_commit.append(rob_entry)
            else:
                break

        for entry in to_commit:
            # Actually commit the architectural register value
            if entry.dest_reg is not None and entry.dest_reg != 0:
                # Move the final value from phys register to arch register
                self.phys_regs[entry.dest_prn] = entry.value
                # In a real design, we might need to track old mappings, etc.
            entry.done = True
        
        # remove committed from the front of the ROB
        while self.ROB and self.ROB[0].done:
            self.ROB.pop(0)
            # Re-index so that self.ROB[i].id = i might remain consistent
            for j, re in enumerate(self.ROB):
                re.id = j



    def _reg_index(self, reg_str):

        reg_str = reg_str.replace('R', '').replace('F', '')
        try:
            return int(reg_str)
        except ValueError:
            return 0  # fallback

    def allocate_physical_register(self):
        """
        Remove one from the free list if available, else stall the pipeline.
        """
        if self.free_list:
            return self.free_list.popleft()
        else:
            # For now, assume we do not handle further stalling logic here,
            # but in a real design, we'd stall decode or rename.
            self.stall_rob_full += 1
            return None

    def has_free_station(self, unit_type):
        # INT -> 4, LOAD -> 4, STORE -> 4, FPADD -> 3, FPMUL -> 3, FPDIV -> 2, BR -> 2
        capacity_map = {
            'INT':   4,
            'LOAD':  2,
            'STORE': 2,
            'FPADD': 3,
            'FPMUL': 3,
            'FPDIV': 2,
            'BR':    2,
        }
        current_len = len(self.res_stations[unit_type])
        return (current_len < capacity_map[unit_type])

    def reserve_station(self, unit_type, instr):
        """
        Create a station entry for `instr` in the specified functional unit queue.
        """
        station_entry = {
            'instr': instr,
            'remaining_lat': LATENCIES[unit_type],
            'busy': True,
            'operands_ready': False,
            'dest_rob_id': None,
        }
        self.res_stations[unit_type].append(station_entry)

    def get_unit_type(self, opcode_str):

        if opcode_str in ('ADD', 'ADDI', 'SLT'):
            return 'INT'
        elif opcode_str in ('FADD', 'FSUB'):
            return 'FPADD'
        elif opcode_str == 'FMUL':
            return 'FPMUL'
        elif opcode_str == 'FDIV':
            return 'FPDIV'
        elif opcode_str == 'FLD':
            return 'LOAD'
        elif opcode_str == 'FSD':
            return 'STORE'
        elif opcode_str == 'BNE':
            return 'BR'
        else:
            return 'INT'  # fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progfile", default='./prog.dat',help="Name of the input file, e.g. prog.dat")
    parser.add_argument("--nf", type=int, default=DEFAULT_NF,
                        help="Number of instructions to fetch per cycle")
    parser.add_argument("--ni", type=int, default=DEFAULT_NI,
                        help="Instruction queue size")
    parser.add_argument("--nw", type=int, default=DEFAULT_NW,
                        help="Issue width")
    parser.add_argument("--nr", type=int, default=DEFAULT_NR,
                        help="Number of ROB entries")
    parser.add_argument("--nb", type=int, default=DEFAULT_NB,
                        help="Number of CDB lines")
    args = parser.parse_args()

    # Read program lines
    with open(args.progfile, 'r') as f:
        program_lines = f.readlines()

    # Instantiate the simulator with provided arguments
    sim = Simulator(args.nf, args.ni, args.nw, args.nr, args.nb, program_lines)

    # Run it
    sim.run()

if __name__ == "__main__":
    main()

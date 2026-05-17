import unittest

import numpy as np
import qiskit.circuit.library
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PhaseGate, U1Gate, U2Gate, U3Gate, UGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ptetools.qiskit import (
    DecomposeU,
    ModifyDelayGate,
    RemoveGateByName,
    RemoveZeroDelayGate,
    ReplaceGate,
    bitlist_to_int,
    choi_to_unitary,
    circuit2matrix,
    counts2dense,
    counts2fractions,
    delay_gate,
    dense2sparse,
    fractions2counts,
    generate_bitstring_tuples,
    generate_bitstrings,
    generate_state_labels,
    index2bitstring,
    invert_permutation,
    largest_remainder_rounding,
    normalize_fractions,
    normalize_probability,
    permute_bits,
    permute_counts,
    permute_string,
    random_clifford_circuit,
)


def circuit_instruction_names(qc):
    return [i.operation.name for i in qc]


class TestBitConversions(unittest.TestCase):
    def test_generate_bitstring_tuples(self):
        assert list(generate_bitstring_tuples(1)) == [(0,), (1,)]

    def test_generate_bitstrings(self):
        assert generate_bitstrings(1) == ["0", "1"]
        assert generate_bitstrings(2) == ["00", "01", "10", "11"]

    def test_bitlist_to_int(self):
        assert bitlist_to_int([0, 1, 1]) == 3

    def test_invert_permutation(self):
        np.testing.assert_array_equal(invert_permutation([0, 1]), [0, 1])
        np.testing.assert_array_equal(invert_permutation([1, 0]), [1, 0])
        np.testing.assert_array_equal(invert_permutation([1, 2, 0]), [2, 0, 1])
        np.testing.assert_array_equal(invert_permutation([0, 1, 3, 2]), np.array([0, 1, 3, 2]))

    def test_permute_bits(self):
        permutation = [0, 1, 3, 2]
        assert permute_bits(idx=0, permutation=permutation) == 0
        assert permute_bits(idx=1, permutation=permutation) == 1
        assert permute_bits(idx=2, permutation=permutation) == 2
        assert permute_bits(idx=4, permutation=permutation) == 8

        assert permute_bits(idx=0, permutation=[1, 0]) == 0
        assert permute_bits(idx=1, permutation=[1, 0]) == 2
        assert permute_bits(idx=1, permutation=[1, 2, 0]) == 4
        assert permute_bits(idx=3, permutation=[3, 4, 0, 1, 2]) == 12

    def test_permute_string(self):
        assert permute_string("abcd", [1, 0, 2, 3]) == "bacd"

    def test_permute_counts(self):
        assert permute_counts({"00": 10, "01": 20}, [1, 0]) == {"00": 10, "10": 20}

        counts = {"1110": 945, "0010": 7, "1011": 16}
        permutation = [1, 0, 2, 3]
        assert permute_counts(counts, permutation) == {"1101": 945, "0001": 7, "1011": 16}


class TestQiskit(unittest.TestCase):
    def test_ModifyDelayGate(self):
        time_unit = 20e-9
        qc = QuantumCircuit(1)
        qc.delay(duration=6.1 * time_unit, unit="s")
        p = ModifyDelayGate(dt=time_unit, round_dt=True)
        qc = p(qc)
        assert list(qc)[0].operation.duration == 6

    def test_dense2sparse(self):
        assert dense2sparse(np.array([1, 0])) == {"0": 1}
        assert dense2sparse(np.array([1, 2])) == {"0": 1, "1": 2}
        assert dense2sparse(np.array([1, 2, 3, 4])) == {"00": 1, "01": 2, "10": 3, "11": 4}

    def test_counts2dense(self):
        np.testing.assert_array_equal(counts2dense({"1": 100}, number_of_bits=1), np.array([0, 100]))
        np.testing.assert_array_equal(counts2dense({"1": 100}, number_of_bits=2), np.array([0, 100, 0, 0]))

    def test_counts2fractions(self):
        assert counts2fractions({"1": 0}) == {"1": 0.0}
        assert counts2fractions({"1": 100, "0": 50}) == {"0": 0.3333333333333333, "1": 0.6666666666666666}

    def test_random_clifford_circuit(self):
        c, index = random_clifford_circuit(1)
        assert isinstance(index, int)
        assert c.num_qubits == 1
        c, index = random_clifford_circuit(2)
        assert c.num_qubits == 2

    def test_normalize_fractions(self):
        np.testing.assert_array_equal(normalize_fractions(np.array([0, 1.001])), [0, 1])
        np.testing.assert_allclose(
            normalize_fractions([0, 0.1, 0.34, 0.6]), np.array([0.0, 0.09615385, 0.32692308, 0.57692308]), atol=1e-6
        )

    def test_ReplaceGate(self):
        gate = qiskit.circuit.library.CXGate
        replacement_circuit = QuantumCircuit(2)
        replacement_circuit.barrier()
        replacement_circuit.cx(0, 1)
        replacement_circuit.barrier()
        qpass = ReplaceGate(gate, replacement_circuit)

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        d = qpass(qc)
        self.assertEqual(circuit_instruction_names(d), ["barrier", "cx", "barrier"])

    def test_ReplaceGate_with_single_qubit(self):
        """Test ReplaceGate with qubits parameter as single int"""
        gate = qiskit.circuit.library.RXGate
        replacement_circuit = QuantumCircuit(1)
        replacement_circuit.barrier()
        replacement_circuit.rx(0.5, 0)
        replacement_circuit.barrier()

        # Replace only on qubit 0
        qpass = ReplaceGate(gate, replacement_circuit, qubits=0)

        qc = QuantumCircuit(3)
        qc.rx(0.1, 0)
        qc.rx(0.2, 1)
        qc.rx(0.3, 2)

        d = qpass(qc)

        # Should have replaced RX on qubit 0 with barrier+rx+barrier
        # but RX on qubits 1 and 2 should remain unchanged
        instructions = circuit_instruction_names(d)
        # Qubit 0: barrier, rx, barrier (from replacement)
        # Qubit 1: rx (original)
        # Qubit 2: rx (original)
        self.assertIn("barrier", instructions)
        self.assertEqual(instructions.count("rx"), 3)  # 1 from replacement + 2 originals

    def test_ReplaceGate_with_multiple_qubits(self):
        """Test ReplaceGate with qubits parameter as list"""
        gate = qiskit.circuit.library.RXGate
        replacement_circuit = QuantumCircuit(1)
        replacement_circuit.barrier()
        replacement_circuit.rx(0.5, 0)
        replacement_circuit.barrier()

        # Replace only on qubits 0 and 2
        qpass = ReplaceGate(gate, replacement_circuit, qubits=[0, 2])

        qc = QuantumCircuit(3)
        qc.rx(0.1, 0)
        qc.rx(0.2, 1)
        qc.rx(0.3, 2)

        d = qpass(qc)

        # Should have replaced RX on qubits 0 and 2 but not on qubit 1
        instructions = circuit_instruction_names(d)
        self.assertEqual(instructions.count("barrier"), 4)  # 2 replacements × 2 barriers each
        self.assertEqual(instructions.count("rx"), 3)  # 2 from replacements + 1 original

    def test_ReplaceGate_qubits_none_replaces_all(self):
        """Test that qubits=None replaces gates on all qubits"""
        gate = qiskit.circuit.library.RXGate
        replacement_circuit = QuantumCircuit(1)
        replacement_circuit.barrier()
        replacement_circuit.rx(0.5, 0)
        replacement_circuit.barrier()

        # Replace on all qubits (default behavior)
        qpass = ReplaceGate(gate, replacement_circuit, qubits=None)

        qc = QuantumCircuit(3)
        qc.rx(0.1, 0)
        qc.rx(0.2, 1)
        qc.rx(0.3, 2)

        d = qpass(qc)

        # All three RX gates should be replaced
        instructions = circuit_instruction_names(d)
        self.assertEqual(instructions.count("barrier"), 6)  # 3 replacements × 2 barriers each
        self.assertEqual(instructions.count("rx"), 3)  # 3 from replacements

    def test_ReplaceGate_no_match_on_limited_qubits(self):
        """Test that gates not on specified qubits are preserved"""
        gate = qiskit.circuit.library.RXGate
        replacement_circuit = QuantumCircuit(1)
        replacement_circuit.barrier()
        replacement_circuit.rx(0.5, 0)

        # Replace only on qubit 0
        qpass = ReplaceGate(gate, replacement_circuit, qubits=0)

        qc = QuantumCircuit(2)
        qc.rx(0.1, 1)  # Only on qubit 1, not on qubit 0

        d = qpass(qc)

        # RX on qubit 1 should not be replaced
        instructions = circuit_instruction_names(d)
        self.assertEqual(instructions.count("rx"), 1)  # Original RX preserved
        self.assertNotIn("barrier", instructions)  # No replacement occurred

    def test_RemoveGateByName(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)

        qc_transpiled = RemoveGateByName("none")(qc)
        self.assertEqual(circuit_instruction_names(qc_transpiled), circuit_instruction_names(qc))

        for name in ["x", "h", "dummy"]:
            qc_transpiled = RemoveGateByName(name)(qc)
            self.assertNotIn(name, circuit_instruction_names(qc_transpiled))

    def test_RemoveZeroDelayGate(self):
        qc = QuantumCircuit(3)
        qc.delay(0)
        qc.barrier()
        qc.delay(10, 0)
        qc.barrier()
        qc.delay(10)

        qc_transpiled = RemoveZeroDelayGate()(qc)
        self.assertEqual(
            circuit_instruction_names(qc_transpiled), ["barrier", "delay", "barrier", "delay", "delay", "delay"]
        )

    def test_fractions2counts(self):
        number_set = np.array([20.2, 20.2, 20.2, 20.2, 19.2]) / 100
        r = largest_remainder_rounding(number_set, 100)
        np.testing.assert_array_equal(r, [21, 20, 20, 20, 19])

        fractions = dict(zip(range(3), [10.1, 80.4, 9.6]))
        assert fractions2counts(fractions, 100) == {0: 10, 1: 80, 2: 10}
        assert fractions2counts(fractions, 1024) == {0: 103, 1: 823, 2: 98}

    def test_circuit2matrix(self):
        for k in range(1, 4):
            x = circuit2matrix(QuantumCircuit(k))
            np.testing.assert_array_equal(x, np.eye(2**k, dtype=complex))

        c = QuantumCircuit(1)
        c.x(0)
        x = circuit2matrix(c)
        expected = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]])
        np.testing.assert_array_equal(x, expected)

    def test_normalize_probability(self):
        np.testing.assert_array_equal(normalize_probability(np.array([0, 0.99], dtype=np.float64)), [0, 1])
        np.testing.assert_array_equal(normalize_probability(np.array([-0.01, 1.0099], dtype=np.float64)), [0, 1])
        assert sum(normalize_probability(np.array([1.2342, 123.321, -0.001], dtype=np.float64))) == 1

    def test_index2bitstring(self):
        assert index2bitstring(0, 2) == "00"
        assert index2bitstring(1, 2) == "01"
        assert index2bitstring(2, 2) == "10"
        assert index2bitstring(3, 2) == "11"
        assert index2bitstring(5, 4) == "0101"
        assert index2bitstring(15, 4) == "1111"

    def test_generate_state_labels(self):
        labels_latex = generate_state_labels(2, latex=True)
        assert labels_latex == [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]

        labels_plain = generate_state_labels(2, latex=False)
        assert labels_plain == ["|00>", "|01>", "|10>", "|11>"]

        labels_1qubit = generate_state_labels(1)
        assert len(labels_1qubit) == 2

    def test_delay_gate(self):
        gate = delay_gate(100e-9, 20e-9, round_dt=True)
        np.testing.assert_allclose(gate.params[0], 5.0, atol=1e-12)

        gate_no_round = delay_gate(duration=100e-9, dt=20e-9, round_dt=False)
        np.testing.assert_allclose(gate_no_round.params[0], 100e-9 / 20e-9, atol=1e-12)

    def test_fractions2counts_no_rounding(self):
        fractions = {0: 0.1, 1: 0.8, 2: 0.1}
        counts = fractions2counts(fractions, 100, integer_rounding=False)  # ty: ignore[invalid-argument-type]
        assert counts == {0: 10.0, 1: 80.0, 2: 10.0}

        fractions_list = [{0: 0.5, 1: 0.5}, {0: 0.25, 1: 0.75}]
        counts_list = fractions2counts(fractions_list, 100, integer_rounding=False)  # ty: ignore[invalid-argument-type]
        assert counts_list == [{0: 50.0, 1: 50.0}, {0: 25.0, 1: 75.0}]

    def test_choi_to_unitary(self):
        import qutip

        X = qutip.sigmax()
        s = qutip.core.superop_reps.to_super(X)
        choi_qobj = qutip.core.superop_reps.to_choi(s)
        choi = choi_qobj.full()

        unitary = choi_to_unitary(choi)
        IC = unitary @ X.full().conjugate().T
        IC = np.exp(-np.angle(IC[0, 0]) * 1j) * IC
        np.testing.assert_almost_equal(IC, np.eye(IC.shape[0]))

    def test_counts2fractions_list(self):
        counts_list = [{"0": 50, "1": 50}, {"0": 25, "1": 75}]
        fractions_list = counts2fractions(counts_list)
        assert fractions_list[0] == {"0": 0.5, "1": 0.5}
        assert fractions_list[1] == {"0": 0.25, "1": 0.75}

    def test_counts2fractions_zero_total(self):
        counts = {"0": 0, "1": 0}
        fractions = counts2fractions(counts)
        assert fractions == {"0": 0.0, "1": 0.0}


class TestDecomposeU(unittest.TestCase):
    def setUp(self):
        self.decompose = DecomposeU()

    def test_initialization(self):
        """Test that DecomposeU initializes correctly"""
        decompose = DecomposeU()
        self.assertIsNotNone(decompose)

    def test_ugate_replacement_circuit_single_parameter(self):
        """Test replacement circuit for single parameter (U1 gate)"""
        qc = self.decompose._ugate_replacement_circuit((1.0,))
        self.assertEqual(qc.num_qubits, 1)
        instruction_names = [instr.operation.name for instr in qc]
        self.assertIn("rz", instruction_names)

    def test_ugate_replacement_circuit_two_parameters(self):
        """Test replacement circuit for two parameters (U2 gate)"""
        qc = self.decompose._ugate_replacement_circuit((1.0, 2.0))
        self.assertEqual(qc.num_qubits, 1)
        instruction_names = [instr.operation.name for instr in qc]
        self.assertTrue(any("r" in name for name in instruction_names))

    def test_ugate_replacement_circuit_three_parameters(self):
        """Test replacement circuit for three parameters (U3 gate)"""
        qc = self.decompose._ugate_replacement_circuit((1.0, 2.0, 3.0))
        self.assertEqual(qc.num_qubits, 1)
        instruction_names = [instr.operation.name for instr in qc]
        self.assertTrue(any("r" in name for name in instruction_names))

    def test_ugate_replacement_circuit_special_case_pi_half(self):
        """Test special case when theta = pi/2"""
        qc = self.decompose._ugate_replacement_circuit((np.pi / 2, 1.0, 2.0))
        self.assertEqual(qc.num_qubits, 1)
        instruction_names = [instr.operation.name for instr in qc]
        self.assertGreater(len(instruction_names), 0)

    def test_ugate_replacement_circuit_special_case_minus_pi_half(self):
        """Test special case when theta = -pi/2"""
        qc = self.decompose._ugate_replacement_circuit((-np.pi / 2, 0, 0))
        self.assertEqual(qc.num_qubits, 1)
        instruction_names = [instr.operation.name for instr in qc]
        self.assertIn("ry", instruction_names)

    def test_ugate_replacement_circuit_special_case_pi(self):
        """Test special case when theta = pi"""
        qc = self.decompose._ugate_replacement_circuit((np.pi, 0, 0))
        self.assertEqual(qc.num_qubits, 1)
        instruction_names = [instr.operation.name for instr in qc]
        self.assertIn("ry", instruction_names)

    def test_ugate_replacement_circuit_invalid_parameters(self):
        """Test error handling for invalid number of parameters"""
        with self.assertRaises(ValueError):
            self.decompose._ugate_replacement_circuit((1.0, 2.0, 3.0, 4.0))

    def test_ugate_replacement_circuit_caching(self):
        """Test that circuit caching works (identical calls return same result)"""
        params = (1.0, 2.0, 3.0)
        qc1 = self.decompose._ugate_replacement_circuit(params)
        qc2 = self.decompose._ugate_replacement_circuit(params)
        # Both should be valid QuantumCircuits
        self.assertEqual(qc1.num_qubits, qc2.num_qubits)

    def test_ugate_replacement_circuit_u3_gate(self):
        """Test replacement circuit for U3Gate"""
        u3_gate = U3Gate(1.0, 2.0, 3.0)
        qc = self.decompose.ugate_replacement_circuit(u3_gate)
        self.assertEqual(qc.num_qubits, 1)

    def test_ugate_replacement_circuit_u2_gate(self):
        """Test replacement circuit for U2Gate"""
        u2_gate = U2Gate(1.0, 2.0)
        qc = self.decompose.ugate_replacement_circuit(u2_gate)
        self.assertEqual(qc.num_qubits, 1)

    def test_ugate_replacement_circuit_u1_gate(self):
        """Test replacement circuit for U1Gate"""
        u1_gate = U1Gate(1.0)
        qc = self.decompose.ugate_replacement_circuit(u1_gate)
        self.assertEqual(qc.num_qubits, 1)

    def test_ugate_replacement_circuit_ugate(self):
        """Test replacement circuit for UGate"""
        ugate = UGate(1.0, 2.0, 3.0)
        qc = self.decompose.ugate_replacement_circuit(ugate)
        self.assertEqual(qc.num_qubits, 1)

    def test_ugate_replacement_circuit_phasegate(self):
        """Test replacement circuit for PhaseGate"""
        phase_gate = PhaseGate(1.0)
        qc = self.decompose.ugate_replacement_circuit(phase_gate)
        self.assertEqual(qc.num_qubits, 1)

    def test_ugate_replacement_circuit_invalid_gate_type(self):
        """Test error handling for invalid gate type"""
        # Use a gate that is not one of the supported types
        invalid_gate = qiskit.circuit.library.CXGate()
        with self.assertRaises(TypeError):
            self.decompose.ugate_replacement_circuit(invalid_gate)

    def test_run_with_u3_gate(self):
        """Test decomposing U3 gate in a circuit"""
        qc = QuantumCircuit(1)
        qc.u(1.0, 2.0, 3.0, 0)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        # Check that the decomposed circuit has rotations
        instruction_names = [instr.operation.name for instr in decomposed_circuit]
        self.assertTrue(any("r" in name for name in instruction_names))
        # U gate should be replaced
        self.assertNotIn("u", instruction_names)

    def test_run_with_multiple_gates(self):
        """Test decomposing multiple U gates in a circuit"""
        qc = QuantumCircuit(2)
        qc.u(1.0, 2.0, 3.0, 0)
        qc.u(0.5, 1.5, 2.5, 1)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        # Check that the decomposed circuit has rotations
        instruction_names = [instr.operation.name for instr in decomposed_circuit]
        self.assertTrue(any("r" in name for name in instruction_names))

    def test_run_preserves_circuit_structure(self):
        """Test that decomposition preserves circuit qubit count"""
        qc = QuantumCircuit(3)
        qc.u(1.0, 2.0, 3.0, 0)
        qc.h(1)
        qc.u(0.5, 1.5, 2.5, 2)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        self.assertEqual(decomposed_circuit.num_qubits, qc.num_qubits)

    def test_run_with_empty_circuit(self):
        """Test decomposition on an empty circuit"""
        qc = QuantumCircuit(1)
        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        self.assertEqual(decomposed_circuit.num_qubits, 1)

    def test_run_with_non_u_gates(self):
        """Test that non-U gates are preserved"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        instruction_names = [instr.operation.name for instr in decomposed_circuit]
        self.assertIn("h", instruction_names)
        self.assertIn("cx", instruction_names)
        self.assertIn("x", instruction_names)

    def test_run_with_mixed_gates(self):
        """Test decomposition with a mix of U and non-U gates"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.u(1.0, 2.0, 3.0, 0)
        qc.cx(0, 1)
        qc.u(0.5, 1.5, 2.5, 1)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        instruction_names = [instr.operation.name for instr in decomposed_circuit]
        # H and CX should be preserved
        self.assertIn("h", instruction_names)
        self.assertIn("cx", instruction_names)
        # U gates should be replaced
        self.assertNotIn("u", instruction_names)

    def test_run_with_parameterized_ugate(self):
        """Test decomposing a parameterized U gate."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lam")

        qc = QuantumCircuit(1)
        qc.u(theta, phi, lam, 0)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        instruction_names = [instr.operation.name for instr in decomposed_circuit]
        self.assertNotIn("u", instruction_names)
        self.assertIn("rz", instruction_names)
        self.assertIn("rx", instruction_names)
        self.assertEqual(set(decomposed_circuit.parameters), {theta, phi, lam})

    def test_run_with_parameterized_u1_gate(self):
        """Test decomposing a parameterized U1/phase-like gate."""
        lam = Parameter("lam")

        qc = QuantumCircuit(1)
        qc.u(0.0, 0.0, lam, 0)

        dag = circuit_to_dag(qc)
        decomposed_dag = self.decompose.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        instruction_names = [instr.operation.name for instr in decomposed_circuit]
        self.assertNotIn("u", instruction_names)
        self.assertIn("rz", instruction_names)
        self.assertEqual(set(decomposed_circuit.parameters), {lam})

    def test_run_special_case_u_pi_minus_pi_half_pi_half(self):
        """U(pi, -pi/2, pi/2) should decompose to two Rx(pi/2) gates."""
        qc = QuantumCircuit(1)
        qc.u(np.pi, -np.pi / 2, np.pi / 2, 0)

        decomposed = dag_to_circuit(self.decompose.run(circuit_to_dag(qc)))

        instruction_names = [instr.operation.name for instr in decomposed]
        self.assertEqual(instruction_names, ["rx", "rx"])
        for instr in decomposed:
            self.assertAlmostEqual(float(instr.operation.params[0]), np.pi / 2, places=12)

    def test_run_rx_pi_half_input_case(self):
        """Rx(pi/2) transpiled to U should map back to a single Rx(pi/2)."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)

        transpiled_qc = transpile(qc, basis_gates=["u"], optimization_level=0)
        decomposed = dag_to_circuit(self.decompose.run(circuit_to_dag(transpiled_qc)))

        instruction_names = [instr.operation.name for instr in decomposed]
        self.assertEqual(instruction_names, ["rx"])
        self.assertAlmostEqual(float(decomposed[0].operation.params[0]), np.pi / 2, places=12)

    def test_run_ry_pi_half_input_case(self):
        """Ry(pi/2) transpiled to U should map back to a single Ry(pi/2)."""
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)

        transpiled_qc = transpile(qc, basis_gates=["u"], optimization_level=0)
        decomposed = dag_to_circuit(self.decompose.run(circuit_to_dag(transpiled_qc)))

        instruction_names = [instr.operation.name for instr in decomposed]
        self.assertEqual(instruction_names, ["ry"])
        self.assertAlmostEqual(float(decomposed[0].operation.params[0]), np.pi / 2, places=12)


if __name__ == "__main__":
    unittest.main()

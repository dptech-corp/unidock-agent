import argparse
import pytest

from unidock_processing.cli.docking import CLICommand


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    CLICommand.add_arguments(p)
    return p


class TestCLIAddArguments:
    def test_parser_has_receptor_argument(self, parser):
        args = parser.parse_args(["-r", "receptor.pdb"])
        assert args.receptor == "receptor.pdb"

    def test_parser_has_ligand_argument(self, parser):
        args = parser.parse_args(["-l", "ligand.sdf"])
        assert args.ligand == "ligand.sdf"

    def test_parser_has_ligand_batch_argument(self, parser):
        args = parser.parse_args(["-lb", "batch.txt"])
        assert args.ligand_batch == "batch.txt"

    def test_parser_has_center_argument(self, parser):
        args = parser.parse_args(["-c", "1.0", "2.0", "3.0"])
        assert args.center == [1.0, 2.0, 3.0]

    def test_center_default(self, parser):
        args = parser.parse_args([])
        assert args.center == [0.0, 0.0, 0.0]

    def test_parser_has_output_argument(self, parser):
        args = parser.parse_args(["-o", "output.sdf"])
        assert args.output_docking_pose_sdf_file_name == "output.sdf"

    def test_output_default(self, parser):
        args = parser.parse_args([])
        assert args.output_docking_pose_sdf_file_name == "unidock2_pose.sdf"

    def test_parser_has_configurations_argument(self, parser):
        args = parser.parse_args(["-cf", "config.yaml"])
        assert args.configurations == "config.yaml"

    def test_parser_has_decomp_flag(self, parser):
        args = parser.parse_args(["--decomp"])
        assert args.decomp is True

    def test_decomp_default_false(self, parser):
        args = parser.parse_args([])
        assert args.decomp is False

    def test_parser_has_ff_minimize_steps(self, parser):
        args = parser.parse_args(["--ff_minimize_steps", "500"])
        assert args.ff_minimize_steps == 500

    def test_ff_minimize_steps_default(self, parser):
        args = parser.parse_args([])
        assert args.ff_minimize_steps == 0

    def test_all_defaults_no_args(self, parser):
        args = parser.parse_args([])
        assert args.receptor is None
        assert args.ligand is None
        assert args.ligand_batch is None
        assert args.configurations is None

    def test_combined_args(self, parser):
        args = parser.parse_args([
            "-r", "protein.pdb",
            "-l", "mol.sdf",
            "-c", "5.0", "10.0", "15.0",
            "-o", "result.sdf",
            "--decomp",
            "--ff_minimize_steps", "200",
        ])
        assert args.receptor == "protein.pdb"
        assert args.ligand == "mol.sdf"
        assert args.center == [5.0, 10.0, 15.0]
        assert args.output_docking_pose_sdf_file_name == "result.sdf"
        assert args.decomp is True
        assert args.ff_minimize_steps == 200

    def test_long_form_args(self, parser):
        args = parser.parse_args([
            "--receptor", "rec.pdb",
            "--ligand", "lig.sdf",
            "--ligand_batch", "batch.txt",
        ])
        assert args.receptor == "rec.pdb"
        assert args.ligand == "lig.sdf"
        assert args.ligand_batch == "batch.txt"

import json
import os
import re
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import zipfile
from abc import ABC, abstractmethod


class PDBParser:
    """Handles PDB file parsing and atom extraction."""

    @staticmethod
    def parse_atom(line: str) -> Tuple[str, int, Tuple[float, float, float], float]:
        """Extract (chain, resSeq, coords, plddt) from an ATOM/HETATM line."""
        return (
            line[21].strip(),  # chain ID
            int(line[22:26]),  # resSeq
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),  # X, Y, Z
            float(line[60:66]),  # pLDDT in B-factor
        )

    @staticmethod
    def get_structure_type(pdb_lines: List[str]) -> str:
        """Determine if structure is 'apo' or 'complex'."""
        chains = set()
        for line in pdb_lines:
            if line.startswith(("ATOM  ", "HETATM")):
                if len(line) <= 21:
                    continue
                chain = line[21].strip()
                chains.add(chain)
        return "apo" if len(chains) == 1 else "complex"

    def get_peptide_chain_info(self, pdb_lines: List[str]) -> Tuple[List[int], str]:
        """Get global indices and chain ID for peptide chain.
        In all complexes from ColabFold, peptide is last chain alphabetically."""
        chains_order = []
        per_chain_res_list = {}
        per_chain_seen = {}

        for line in pdb_lines:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) <= 21:
                continue
            try:
                parsed = self.parse_atom(line)
                res_seq, chain = parsed[1], parsed[0]

                if chain not in per_chain_res_list:
                    chains_order.append(chain)
                    per_chain_res_list[chain] = []
                    per_chain_seen[chain] = set()

                if res_seq not in per_chain_seen[chain]:
                    per_chain_res_list[chain].append(res_seq)
                    per_chain_seen[chain].add(res_seq)
            except Exception:
                continue

        if not chains_order:
            raise ValueError("No chains found in PDB lines")

        peptide_chain = sorted(chains_order)[-1]

        # Build global index mapping
        global_index = 1
        global_map = {}
        for chain in chains_order:
            for res in per_chain_res_list[chain]:
                global_map[(chain, res)] = global_index
                global_index += 1

        peptide_residues = per_chain_res_list[peptide_chain]
        peptide_global_indices = [
            global_map[(peptide_chain, r)] for r in peptide_residues
        ]

        return peptide_global_indices, peptide_chain


class MetricsCalculator:
    """Calculates various structural quality metrics."""

    def __init__(self):
        self.pdb_parser = PDBParser()

    @staticmethod
    def average(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def calculate_peptide_plddt(
        self, pdb_lines: List[str], json_data: Dict, round_digits: int = 2
    ) -> Tuple[float, str]:
        """Calculate average pLDDT for peptide chain."""
        peptide_indices, peptide_chain = self.pdb_parser.get_peptide_chain_info(
            pdb_lines
        )
        plddt_values = json_data.get("plddt", [])

        if not peptide_indices:
            raise ValueError(f"No residues found for chain '{peptide_chain}'")

        peptide_plddt = [
            plddt_values[i - 1] for i in peptide_indices if 0 < i <= len(plddt_values)
        ]

        if not peptide_plddt:
            raise ValueError(f"No pLDDT values found for chain '{peptide_chain}'")

        return round(self.average(peptide_plddt), round_digits), peptide_chain

    def calculate_interface_plddt(
        self,
        pdb_lines: List[str],
        chain_protein: str = "A",
        chain_peptide: str = "B",
        distance_cutoff: float = 5.0,
        round_digits: int = 2,
    ) -> Dict:
        """Calculate interface pLDDT between protein and peptide chains."""
        protein_atoms, peptide_atoms = [], []

        for line in pdb_lines:
            if line.startswith(("ATOM  ", "HETATM")):
                chain, resi, xyz, plddt = self.pdb_parser.parse_atom(line)
                if chain == chain_protein:
                    protein_atoms.append((resi, xyz, plddt))
                elif chain == chain_peptide:
                    peptide_atoms.append((resi, xyz, plddt))

        if not protein_atoms or not peptide_atoms:
            raise ValueError("Could not find both chains in the PDB")

        d2_cut = distance_cutoff**2
        prot_if_residues, pep_if_residues = set(), set()
        prot_if_plddt, pep_if_plddt = [], []

        for resi_b, xyz_b, plddt_b in peptide_atoms:
            xb, yb, zb = xyz_b
            for resi_a, xyz_a, plddt_a in protein_atoms:
                xa, ya, za = xyz_a
                if (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2 <= d2_cut:
                    pep_if_residues.add(resi_b)
                    prot_if_residues.add(resi_a)
                    pep_if_plddt.append(plddt_b)
                    prot_if_plddt.append(plddt_a)
                    break

        return {
            "protein_residues": sorted(prot_if_residues),
            "peptide_residues": sorted(pep_if_residues),
            "protein_avg": round(self.average(prot_if_plddt), round_digits),
            "peptide_avg": round(self.average(pep_if_plddt), round_digits),
            "overall_avg": round(
                self.average(prot_if_plddt + pep_if_plddt), round_digits
            ),
        }

    def summarize_plddt(
        self, record: Dict, pdb_lines: List[str], round_digits: int = 2
    ) -> Dict:
        """Summarize pLDDT statistics for the structure."""
        plddt = record.get("plddt", [])
        if not plddt:
            return {"plddt": [0.0, 0.0, 0.0, 0.0]}

        n = len(plddt)
        mean = sum(plddt) / n
        s = sorted(plddt)
        mid = n // 2
        median = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

        struct_type = self.pdb_parser.get_structure_type(pdb_lines)

        if struct_type != "complex":
            stats = [mean, median, 0.0, 0.0]
            return {"plddt": [round(x, round_digits) for x in stats]}
        else:
            peptide_plddt, peptide_chain = self.calculate_peptide_plddt(
                pdb_lines, record, round_digits
            )
            interface_info = self.calculate_interface_plddt(
                pdb_lines, "A", peptide_chain, 5.0, round_digits
            )

            stats = [mean, median, peptide_plddt, interface_info["overall_avg"]]
            return {
                "plddt": [round(x, round_digits) for x in stats],
                "interface": {
                    "prot_interface": interface_info["protein_residues"],
                    "pep_interface": interface_info["peptide_residues"],
                },
            }

    def calculate_ptm_values(
        self, record: Dict, round_digits: Optional[int] = None
    ) -> List:
        """Extract pTM values from record."""
        global_ptm = record.get("ptm", None)
        per_chain = record.get("per_chain_ptm", {}) or {}
        chain_ids = sorted(per_chain.keys())
        values = [global_ptm] + [per_chain[c] for c in chain_ids]

        if round_digits is not None:
            values = [None if v is None else round(v, round_digits) for v in values]
        return values

    def summarize_pae(
        self, record: Dict, threshold: float = 5.0, round_digits: int = 2
    ) -> List[float]:
        """Summarize PAE statistics."""
        max_pae = record.get("max_pae", None)
        pae = record.get("pae", [])

        if not pae or not isinstance(pae, list):
            return [0.0, 0.0, 0.0, 0.0]

        # Flatten for mean/median
        flat = []
        for row in pae:
            if not isinstance(row, list):
                continue
            for x in row:
                if x is None:
                    continue
                try:
                    v = float(x)
                    if v == v and v != float("inf") and v != float("-inf"):
                        flat.append(v)
                except Exception:
                    continue

        if not flat:
            return [0.0, 0.0, 0.0, 0.0]

        mean = sum(flat) / len(flat)
        s = sorted(flat)
        n = len(s)
        mid = n // 2
        median = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

        # Calculate coverage
        row_means = []
        for row in pae:
            vals = []
            if isinstance(row, list):
                for x in row:
                    if x is None:
                        continue
                    try:
                        v = float(x)
                        if v == v and v != float("inf") and v != float("-inf"):
                            vals.append(v)
                    except Exception:
                        continue
            if vals:
                row_means.append(sum(vals) / len(vals))
            else:
                row_means.append(float("nan"))

        # Longest consecutive run below threshold
        longest = current = 0
        for v in row_means:
            if (v == v) and (v < float(threshold)):
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        seq_len = len(row_means) if row_means else 0
        coverage = (longest / seq_len) if seq_len else 0.0

        stats = [max_pae, mean, median, coverage]
        return [round(x, round_digits) for x in stats]


class StructureClassifier(ABC):
    """Abstract base class for structure classification."""

    @abstractmethod
    def classify(self, entry_result: Dict) -> Dict:
        pass


class StandardClassifier(StructureClassifier):
    """Standard classification criteria."""

    def classify(self, entry_result: Dict) -> Dict:
        if entry_result.get("rank001", {}).get("iptm", []) == []:
            return self._classify_apo(entry_result)
        else:
            return self._classify_complex(entry_result)

    def _classify_apo(self, entry_result: Dict) -> Dict:
        data = entry_result.get("rank001", {})
        avg_plddt = data.get("plddt", [0])[0]
        ptm = data.get("ptm", [0])[0]

        if avg_plddt >= 70.0 and ptm >= 0.5:
            entry_result["rank001"]["tier"] = 1
        else:
            entry_result["rank001"]["tier"] = 4

        return entry_result

    def _classify_complex(self, entry_result: Dict) -> Dict:
        data = entry_result.get("rank001", {})
        composite_ptm = round(data.get("composite_ptm", 0), 2)
        iptm = data.get("iptm", 0)
        peptide_plddt = data.get("plddt", [0, 0, 0, 0])[-2]
        interface_plddt = data.get("plddt", [0, 0, 0, 0])[-1]
        actifptm = data.get("actifptm", 0)

        if composite_ptm >= 0.75 and iptm >= 0.8 and peptide_plddt >= 70.0:
            tier = 1
        elif (
            composite_ptm >= 0.75
            and iptm >= 0.8
            and peptide_plddt < 70.0
            and (interface_plddt >= 70.0 or actifptm >= 0.8)
        ):
            tier = 2
        elif (
            0.5 <= composite_ptm < 0.75
            and 0.5 <= iptm < 0.8
            and (interface_plddt >= 70.0 or actifptm >= 0.8)
        ):
            tier = 3
        else:
            tier = 4

        entry_result["rank001"]["tier"] = tier
        return entry_result


class OnTargetClassifier(StructureClassifier):
    """On-target specific classification criteria."""

    def classify(self, entry_result: Dict) -> Dict:
        if entry_result.get("rank001", {}).get("iptm", []) == []:
            return self._classify_apo(entry_result)
        else:
            return self._classify_complex(entry_result)

    def _classify_apo(self, entry_result: Dict) -> Dict:
        data = entry_result.get("rank001", {})
        avg_plddt = data.get("plddt", [0])[0]
        ptm = data.get("ptm", [0])[0]

        if avg_plddt >= 70.0 or ptm >= 0.5:
            entry_result["rank001"]["tier"] = 1
        else:
            entry_result["rank001"]["tier"] = 4

        return entry_result

    def _classify_complex(self, entry_result: Dict) -> Dict:
        data = entry_result.get("rank001", {})
        composite_ptm = round(data.get("composite_ptm", 0), 2)
        iptm = data.get("iptm", 0)
        peptide_plddt = data.get("plddt", [0, 0, 0, 0])[-2]
        interface_plddt = data.get("plddt", [0, 0, 0, 0])[-1]

        if composite_ptm >= 0.51 and (
            iptm >= 0.42 or peptide_plddt >= 42.87 or interface_plddt >= 72.82
        ):
            entry_result["rank001"]["tier"] = 1
        else:
            entry_result["rank001"]["tier"] = 4

        return entry_result


class ColabFoldAnalyzer:
    """Main class for analyzing ColabFold results."""

    def __init__(self, classifier: StructureClassifier = None):
        self.metrics_calculator = MetricsCalculator()
        self.classifier = classifier or StandardClassifier()

    def unzip_colabfold(self, folder: str):
        """Unzip ColabFold output files."""
        for fname in os.listdir(folder):
            if fname.endswith(".zip"):
                zip_path = os.path.join(folder, fname)
                extract_dir = os.path.join(folder, "outputdir")

                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)

                inner_lv1 = os.path.join(extract_dir, "outputdir")
                if os.path.isdir(inner_lv1):
                    inner_items = os.listdir(inner_lv1)
                    if len(inner_items) == 1:
                        inner_src = os.path.join(inner_lv1, inner_items[0])
                        dst = os.path.join(folder, inner_items[0])

                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.move(inner_src, dst)

                shutil.rmtree(extract_dir)

    def get_processing_time(self, log_path: Path) -> Optional[int]:
        """Extract processing time from log file."""
        try:
            with open(log_path) as f:
                lines = f.readlines()

            timestamps = [
                re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
                for line in lines
            ]
            timestamps = [m.group(1) for m in timestamps if m]

            if len(timestamps) < 2:
                return None

            fmt = "%Y-%m-%d %H:%M:%S,%f"
            start = datetime.strptime(timestamps[0], fmt)
            end = datetime.strptime(timestamps[-1], fmt)
            return int(round((end - start).total_seconds()))
        except Exception:
            return None

    def process_rank_file(
        self, json_path: Path, pdb_path: Path, rank001: bool = True
    ) -> Dict:
        """Process a single rank file."""
        with open(json_path, "r") as f, open(pdb_path, "r") as fh:
            rec = json.load(f)
            pdb_lines = fh.readlines()

        iptm = rec.get("iptm", [])
        actifptm = rec.get("actifptm", [])
        global_ptm = rec.get("ptm", None)
        composite_ptm = (
            0.8 * iptm + 0.2 * global_ptm if (iptm and global_ptm is not None) else 0.0
        )

        plddt_summary = self.metrics_calculator.summarize_plddt(rec, pdb_lines)
        interface = plddt_summary.get("interface", {})

        result = {
            "plddt": plddt_summary.get("plddt", [0.0, 0.0, 0.0, 0.0]),
            "ptm": self.metrics_calculator.calculate_ptm_values(rec, round_digits=2),
            "iptm": iptm,
            "composite_ptm": composite_ptm,
            "actifptm": actifptm,
            "pae": self.metrics_calculator.summarize_pae(rec),
        }

        if rank001:
            result["interface"] = interface
        else:
            result["plddt"] = result["plddt"][:3]  # Only mean, median, peptide_plddt

        return result

    def collect_entry(self, entry_dir: Path) -> Dict:
        """Process a single entry directory."""
        entry_result = {}

        try:
            # Find PDB files
            pdb_matches = sorted(entry_dir.glob("*_relaxed_rank_001_*.pdb"))
            if not pdb_matches:
                pdb_matches = sorted(entry_dir.glob(f"{entry_dir.name}.pdb"))

            # Get processing time
            log_matches = sorted(entry_dir.glob("*log.txt"))
            process_time = (
                self.get_processing_time(log_matches[0]) if log_matches else None
            )

            # Get sequence length
            rank1_matches = sorted(entry_dir.glob("*_scores_rank_001_*.json"))
            if not rank1_matches:
                print(f"Warning: No rank 001 JSON file found for {entry_dir.name}")
                return {}

            with open(rank1_matches[0], "r") as f:
                rec = json.load(f)
                length = len(rec.get("plddt", []))

            # Process all ranks
            for i in range(1, 6):
                key = f"rank{i:03d}"
                pattern = entry_dir.glob(f"*_scores_rank_{i:03d}_*.json")
                matches = sorted(pattern)

                if not matches:
                    continue

                try:
                    entry_result[key] = self.process_rank_file(
                        matches[0], pdb_matches[0], rank001=(i == 1)
                    )
                except Exception as e:
                    print(
                        f"Warning: Error processing {entry_dir.name} rank {i}: {str(e)}"
                    )
                    continue

            entry_result["length"] = length
            entry_result["processing_time"] = process_time
            return self.classifier.classify(entry_result)

        except Exception as e:
            print(f"Error processing entry {entry_dir.name}: {str(e)}")
            return {}

    def process_colabfold_output(self, parent_folder: Path) -> Dict:
        """Process all entries in a ColabFold output folder."""
        parent_folder = Path(parent_folder)
        skip_dirs = {"pdb"}
        results = {}

        for entry_path in sorted(parent_folder.iterdir(), key=lambda p: p.name):
            if not entry_path.is_dir() or entry_path.name in skip_dirs:
                continue

            entry_data = self.collect_entry(entry_path)
            if entry_data:
                results[entry_path.name] = entry_data

        return results

    def export_high_quality_structures(
        self, results: Dict, source_dir: Path, output_dir: Path
    ) -> List[str]:
        """Export high-quality structures to output directory."""
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)

        if not source_dir.exists():
            raise FileNotFoundError(f"{source_dir} does not exist")

        output_dir.mkdir(parents=True, exist_ok=True)

        good_entries = []
        bad_entries = []

        for entry_name in sorted(results.keys()):
            tier = results.get(entry_name, {}).get("rank001", {}).get("tier", 0)
            if tier >= 3:
                bad_entries.append(entry_name)
            else:
                good_entries.append(entry_name)

        print(f"High-quality structures: {len(good_entries)}/{len(results)}")
        print(f"Low-quality structures: {len(bad_entries)}/{len(results)}")

        for entry_name in good_entries:
            entry_dir = source_dir / entry_name
            if not entry_dir.exists():
                continue

            pdb_matches = sorted(entry_dir.glob("*_relaxed_rank_001_*.pdb"))
            if pdb_matches:
                pdb_file = pdb_matches[0]
            else:
                pdb_file = entry_dir / f"{entry_name}.pdb"
                if not pdb_file.exists():
                    print(f"Warning: No PDB file found for {entry_name}")
                    continue

            dest_path = output_dir / f"{entry_name}.pdb"
            try:
                shutil.copy(str(pdb_file), str(dest_path))
            except Exception as e:
                print(f"Error copying {pdb_file} -> {dest_path}: {e}")

        return good_entries


# Usage example:
if __name__ == "__main__":
    # Standard analysis
    analyzer = ColabFoldAnalyzer()
    results = analyzer.process_colabfold_output("/path/to/colabfold/output")
    good_entries = analyzer.export_high_quality_structures(
        results, "/path/to/source", "/path/to/output"
    )

    # On-target analysis
    ontarget_analyzer = ColabFoldAnalyzer(OnTargetClassifier())
    ontarget_results = ontarget_analyzer.process_colabfold_output(
        "/path/to/colabfold/output"
    )

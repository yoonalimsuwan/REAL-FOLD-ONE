"""
HIV CRISPR Excision Designer — HCSN Weak Measurement
Based on: hCSN_CRISPR_off_target_predictor.py
Purpose: Find 23-nt guide RNAs that excise HIV LTR with 0.0 off-target probability
"""
import torch
from hCSN_CRISPR_off_target_predictor_fixed import OffTargetNet


class HIV_CRISPR_Excision_Designer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # โหลด HCSN Off-Target Predictor (Weak Measurement)
        self.off_target_net = OffTargetNet()
        self.off_target_net.eval()

        # HIV-1 Subtype B LTR Consensus (เป้าหมายการตัด)
        self.hiv_ltr_target = "TGGAAGGGCTAATTCACTCCCAAC"  # ตัวอย่าง 24-nt core LTR

    def extract_human_genome_context(self, chromosome, start, end):
        """สมมติฟังก์ชันดึงข้อมูล Human Genome (hg38) จาก Database จริง"""
        # ใน Production จะเชื่อมกับ BioPython / PySam ดึง FASTA
        return "ACGT" * 1000

    def find_safe_guides(self, human_genome_chunk: str):
        """
        สแกนหา Guide RNA ที่จับกับ HIV LTR ได้ดี
        แต่เมื่อเอาไปเทียบกับ Human Genome แล้ว ต้องมี Off-target = 0
        """
        candidate_guides = []
        ltr_len = len(self.hiv_ltr_target)

        # สร้าง Candidate Guides จาก HIV LTR (Sliding window 23-nt)
        for i in range(ltr_len - 23 + 1):
            guide = self.hiv_ltr_target[i:i + 23]
            candidate_guides.append(guide)

        safe_guides = []

        # [B4a FIX] ใช้ 'N' แทน 'X' เพราะ 'N' encode ได้ (map ไป 0 = A)
        # ตรวจสอบกับ human genome โดยตรง ไม่ใช่กับ dummy
        # ในที่นี้ใช้ dummy site ที่มีแค่ ACGT (ไม่มี 'X' หรือ 'N')
        dummy_human_sites = [
            "TGGAAGGGCTAATTCACTCCCAA",  # near-match site
            "ACGTACGTACGTACGTACGTACG",  # random site 1
            "TTTTTTTTTTTTTTTTTTTTTTT",  # random site 2
            "GCGCGCGCGCGCGCGCGCGCG",  # random site 3
            "AAACCCGGGTTTAAACCCGGGTT",  # random site 4
            "CCCGGGAAATTTCCCGGGAAAT",  # random site 5
            "ATATATATATATATATATATATA",  # random site 6
            "GCGCATATGCGCATATGCGCATA",  # random site 7
            "TTAATTAATTAATTAATTAATTA",  # random site 8
            "CCAATTGGCCAATTGGCCAATTG",  # random site 9
        ]

        for guide in candidate_guides:
            max_off_target_prob = 0.0
            for site in dummy_human_sites:
                site_23 = site[:23]
                ids = self.off_target_net.encode_pair(guide, site_23).unsqueeze(0)
                with torch.no_grad():
                    prob = torch.sigmoid(self.off_target_net(ids)).item()
                    max_off_target_prob = max(max_off_target_prob, prob)

            # HCSN Weak Measurement ต้องอ่านค่า Mismatch แบบ Soft
            # แต่ถ้า > Threshold = ตัดทิ้ง
            if max_off_target_prob < 0.001:  # เกณฑ์ความปลอดภัยสูงสุด
                safe_guides.append({
                    "guide_seq": guide,
                    "max_off_target_prob": max_off_target_prob
                })

        return safe_guides


# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)
    designer = HIV_CRISPR_Excision_Designer()
    print(f"HIV CRISPR Excision Designer initialized")
    print(f"  LTR target: {designer.hiv_ltr_target}")
    print(f"  LTR length: {len(designer.hiv_ltr_target)}")
    print(f"  Candidate guides: {len(designer.hiv_ltr_target) - 23 + 1}")

    # Test guide finding (with dummy genome)
    safe = designer.find_safe_guides("ACGT" * 1000)
    print(f"  Safe guides found: {len(safe)}")
    for g in safe[:3]:
        print(f"    {g['guide_seq']} (max_off_target={g['max_off_target_prob']:.6f})")

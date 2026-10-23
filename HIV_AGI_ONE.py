from agi_one_v3_11 import EcosystemOrchestrator, AGIONE, AGIConfig
# สมมติว่า import โมดูลที่เราเขียนด้านบนมาเรียบร้อย

def build_hiv_eradication_ecosystem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. สร้าง AGI ONE Core
    agi_cfg = AGIConfig(latent_dim=512, action_dim=64, device=device)
    agi_one = AGIONE(agi_cfg)
    
    # 2. สร้าง Ecosystem Orchestrator
    orchestrator = EcosystemOrchestrator(agi_latent_dim=512, device=device)
    
    # 3. Register HIV Specific Domains
    # Domain: HIV Dynamics (Shock & Kill)
    ltr_engine = HIV_LTR_BistableEngine().to(device)
    orchestrator.register(
        name="hiv_ltr_dynamics", 
        module=ltr_engine, 
        domain="evolution_bv", # ใช้ domain evolution เพราะเป็นพลวัตของไวรัส
        latent_dim=512,
        quality_fn=lambda: 0.95 # สมมติค่า quality จาก CME residual
    )
    
    # Domain: CRISPR Excision (Gene Editing)
    crispr_designer = HIV_CRISPR_Excision_Designer().to(device)
    orchestrator.register(
        name="hiv_crispr_excision",
        module=crispr_designer,
        domain="math", # ใช้ domain math เพราะเป็นการหา Sequence ที่ Exact match
        latent_dim=512
    )
    
    # Domain: Env Trimer Vaccine (Structural Biology)
    # (สมมติว่าโหลด StructuralGNOFold ที่เทรนมาแล้ว)
    # gno_fold = load_pretrained_gno_fold()
    # env_designer = HIV_Env_Immunogen_Designer(gno_fold).to(device)
    # orchestrator.register("hiv_env_vaccine", env_designer, "fold", 512)
    
    # 4. Attach to AGI ONE
    agi_one.attach_orchestrator(orchestrator)
    
    print("✅ HIV Eradication Ecosystem Registered successfully.")
    print(orchestrator.quality_report())
    return agi_one, orchestrator

# รันระบบ
# agi, orch = build_hiv_eradication_ecosystem()

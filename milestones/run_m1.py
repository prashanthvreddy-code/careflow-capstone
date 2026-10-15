"""M1 - Structured intake. Free-text patient messages -> validated PatientRequest objects.

The LLM extracts symptoms (each with an urgency band), the specialty, the insurance ID and
whether the patient asks for clinical advice; code then sets the overall urgency (highest band),
checks the insurance ID really appears in the message, and lists missing fields.
Live: 3 records, 1 LLM call each.   Run:  python milestones/run_m1.py
"""
try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

def _v(x):                     # enum or plain string -> plain string
    return getattr(x, "value", x)


DEMO_IDS = ("CAREFLOW-00001", "CAREFLOW-00002", "CAREFLOW-00004")   # web form, email, phone call


def run(llm, ids: tuple[str, ...] = DEMO_IDS) -> list[dict]:
    from capstone.agents.intake_agent import parse_intake
    from capstone.workflows.graph import load_intake_records

    records = load_intake_records()
    rows = []
    for rid in ids:
        rec = records[rid]
        req = parse_intake(rec, llm)
        truth = rec["ground_truth"]
        rows.append({"record_id": rid, "patient_ref": req.patient_ref, "channel": _v(req.channel),
                     "urgency": _v(req.urgency), "expected_urgency": truth["urgency"],
                     "specialty": _v(req.specialty),
                     "expected_specialty": truth["specialty"],
                     "clinical_advice": req.seeks_clinical_advice,
                     "expected_clinical_advice": truth["seeks_clinical_advice"],
                     "insurance_id_found": req.insurance_id is not None,
                     "missing_fields": req.missing_fields})
    return rows


def show(rows: list[dict]) -> None:
    _common.header("M1 Structured intake", "predicted (answer key)")
    for r in rows:
        print(f"{r['record_id']} {r['patient_ref']} [{r['channel']}]\n"
              f"  urgency   {r['urgency']} ({r['expected_urgency']})\n"
              f"  specialty {r['specialty']} ({r['expected_specialty']})\n"
              f"  clinical advice asked {r['clinical_advice']} ({r['expected_clinical_advice']})\n"
              f"  insurance ID found {r['insurance_id_found']}, missing {r['missing_fields'] or 'none'}")


if __name__ == "__main__":
    show(run(_common.live_llm()))

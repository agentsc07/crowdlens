def cost(n_total, n_verified, sec_per_verify):
    saved = n_total - n_verified
    return {"embedded": n_total, "verified": n_verified, "saved_calls": saved, "saved_seconds": saved * sec_per_verify}

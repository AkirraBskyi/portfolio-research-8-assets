"""CAPM: ожидаемая доходность = rf + beta * ERP."""


def capm_returns(betas, rf, erp):
    return rf + betas * erp

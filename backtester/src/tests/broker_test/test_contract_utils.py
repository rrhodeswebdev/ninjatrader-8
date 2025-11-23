from decimal import Decimal

import pytest
from backintime.broker.futures.contracts import ContractSettings, ContractUtils


@pytest.fixture
def contract_utils() -> ContractUtils:
    settings = ContractSettings(
        per_contract_fee = Decimal('0.62'),
        per_contract_quotient = Decimal('2'),
        per_contract_init_margin = Decimal('1600'),
        per_contract_maintenance_margin = Decimal('1500'),
        per_contract_overnight_margin = Decimal('2200'),
        additional_collateral = Decimal('1400'))

    return ContractUtils(settings)


def test_init_margin(contract_utils):
    sample_usd_amount = Decimal('10_000')
    num_contracts, init_req, fees = contract_utils.estimate_init(sample_usd_amount)
    assert num_contracts == 3
    assert init_req == Decimal('9000') # (init(1600) + additional_collateral(1400)) * 3
    assert fees == Decimal('1.86')  # 0.62*3


def test_init_margin_fees_edge_case(contract_utils):
    sample_usd_amount = Decimal('9_001') # not enough for 3 contracts due 1.86 in fees
    num_contracts, init_req, fees = contract_utils.estimate_init(sample_usd_amount)
    assert num_contracts == 2   # can only afford 2
    assert init_req == Decimal('6000') # (init(1600) + additional_collateral(1400)) * 2
    assert fees == Decimal('1.24')  # 0.62*2


def test_init_margin_net(contract_utils):
    sample_usd_amount = Decimal('9001') # this time it's enough for 3 because we don't care about fees
    num_contracts, init_req, fees = contract_utils.estimate_init_net(sample_usd_amount)
    assert num_contracts == 3
    assert init_req == Decimal('9000') # (init(1600) + additional_collateral(1400)) * 3
    assert fees == Decimal('1.86')  # 0.62*3

'''
def test_maintenance_margin(contract_utils):
    sample_usd_amount = Decimal('9000')
    num_contracts, maintenance_req, fees = contract_utils.estimate_maintenance(sample_usd_amount)
    assert num_contracts == 3
    assert init_req == Decimal('4500') # (init(1600) + additional_collateral(1400)) * 3
    assert fees == Decimal('1.86')  # 0.62*3'''
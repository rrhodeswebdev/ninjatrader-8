from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR


@dataclass
class ContractSettings:
    per_contract_fee: Decimal = Decimal('0.62')
    per_contract_quotient: Decimal = Decimal('2')
    per_contract_init_margin: Decimal = Decimal('1500')
    per_contract_maintenance_margin: Decimal = Decimal('1500')
    per_contract_overnight_margin: Decimal = Decimal('1500')
    additional_collateral: Decimal = Decimal('0')


class ContractUtils:
    def __init__(self, contract_settings: ContractSettings):
        self._settings = contract_settings

    def estimate_init(self, usd_amount):
        """Estimate initial margin requirements (init. margin PLUS additional collateral if set)
        for a maximum amount of contracts for the given `usd_amount`, also considering fee per contract.
        Fees are NOT included in the margin estimation and returned as a separate variable."""
        fee = self._settings.per_contract_fee
        init_margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral

        init_req = init_margin + additional_collateral
        num_contracts = (usd_amount/init_req).quantize(Decimal('1'), ROUND_FLOOR)
        # lower the contracts amount if initial margin PLUS fees exceeds `usd_amount`
        if init_req*num_contracts + fee*num_contracts > usd_amount:
            num_contracts -= 1

        return num_contracts, init_req*num_contracts, fee*num_contracts

    def estimate_init_net(self, usd_amount):
        """Estimate initial margin requirements (init. margin PLUS additional collateral if set)
        for a maximum amount of contracts for the given `usd_amount`, NOT considering fee per contract.
        Fees are NOT included in the margin estimation and returned as a separate variable."""
        fee = self._settings.per_contract_fee
        init_margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral

        init_req = init_margin + additional_collateral
        num_contracts = (usd_amount/init_req).quantize(Decimal('1'), ROUND_FLOOR)

        return num_contracts, init_req*num_contracts, fee*num_contracts

    def estimate_maintenance(self, usd_amount):
        fee = self._settings.per_contract_fee
        init_margin = self._settings.per_contract_init_margin
        maintenance_margin = self._settings.per_contract_maintenance_margin
        additional_collateral = self._settings.additional_collateral

        init_req = init_margin + additional_collateral
        maintenance_req = maintenance_margin + additional_collateral
        num_contracts = (usd_amount/maintenance_req).quantize(Decimal('1'), ROUND_FLOOR)
        if init_req*num_contracts + fee*num_contracts > usd_amount:
            num_contracts -= 1

        return num_contracts, maintenance_req*num_contracts, fee*num_contracts 

    def estimate_maintenance_net(self, usd_amount):
        fee = self._settings.per_contract_fee
        maintenance_margin = self._settings.per_contract_maintenance_margin
        additional_collateral = self._settings.additional_collateral

        margin_req = maintenance_margin + additional_collateral
        num_contracts = (usd_amount/margin_req).quantize(Decimal('1'), ROUND_FLOOR)
        return num_contracts, margin_req*num_contracts, fee*num_contracts

    def estimate(self, usd_amount):
        """Estimate maximum affordable number of contracts to purchase for a given usd_amount, 
        initial margin for the contracts, and trading fee for purchasing them."""
        fee = self._settings.per_contract_fee
        init_margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral
        margin_req = init_margin + additional_collateral

        num_contracts = (usd_amount/margin_req).quantize(Decimal('1'), ROUND_FLOOR)
        if margin_req*num_contracts + fee*num_contracts > usd_amount:
            num_contracts -= 1

        return num_contracts, margin_req*num_contracts, fee*num_contracts

    def estimate_net(self, usd_amount):
        fee = self._settings.per_contract_fee
        init_margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral
        margin_req = init_margin + additional_collateral
        num_contracts = (usd_amount/margin_req).quantize(Decimal('1'), ROUND_FLOOR)
        return num_contracts, margin_req*num_contracts, fee*num_contracts

    def get_contracts(self, usd_amount):    # v1
        maintenance = self._settings.per_contract_maintenance_margin
        additional_collateral = self._settings.additional_collateral
        per_contract_req = maintenance + additional_collateral
        return (usd_amount/per_contract_req).quantize(Decimal('1'), ROUND_FLOOR)
    '''
    def get_contracts(self, usd_amount):
        init_margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral
        per_contract_req = init_margin + additional_collateral
        return (usd_amount/per_contract_req).quantize(Decimal('1'), ROUND_FLOOR)

    def get_collateral(self, num_contracts):
        init_margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral
        per_contract_req = init_margin + additional_collateral
        return num_contracts*per_contract_req'''

    def get_collateral(self, num_contracts):
        margin = self._settings.per_contract_init_margin
        additional_collateral = self._settings.additional_collateral
        per_contract_req = margin + additional_collateral
        return num_contracts*per_contract_req

    def get_init_margin(self, num_contracts):
        """Get the initial margin for a given number of contracts."""
        return num_contracts*self._settings.per_contract_init_margin
    
    def get_maintenance_margin(self, num_contracts):
        """Get the maintenance margin for a given number of contracts."""
        return num_contracts*self._settings.per_contract_maintenance_margin
    
    def get_overnight_margin(self, num_contracts):
        """Get the overnight margin for a given number of contracts."""
        return num_contracts*self._settings.per_contract_overnight_margin

    def get_net_gain(self, fill_price, fills, contracts, long=True):
        """Return net gain and fees using Decimal-safe math to avoid float mixing."""
        net_gain = Decimal('0')
        fees = Decimal('0')

        fee_rate = self._settings.per_contract_fee
        quotient = self._settings.per_contract_quotient

        for entry_fill, num_contracts in zip(fills, contracts):
            # ensure Decimal arithmetic
            fp = Decimal(str(fill_price))
            ef = Decimal(str(entry_fill))
            price_diff = fp - ef if long else ef - fp
            gain = price_diff * Decimal(num_contracts) * quotient
            fee = Decimal(num_contracts) * fee_rate
            fees += fee
            net_gain += gain - fee
        return net_gain, fees

import typing as t
from decimal import Decimal
from dataclasses import dataclass
from itertools import count
from collections import OrderedDict


class EmptyPosition(Exception):
    def __init__(self):
        super().__init__("Empty position")


class NotEnoughInPosition(Exception):
    def __init__(self, req_amount, actual_amount):
        msg = (f"Not enough in position: need {req_amount} "
               f"but only have {actual_amount}")
        super().__init__(msg)


class ReleaseMoreThanAcquired(Exception):
    def __init__(self, req_amount, acquired_amount):
        msg = (f"Can not release more than acquired. "
               f"Requested {req_amount} but acquired {acquired_amount}")
        super().__init__(msg)


class TpSlNotFound(Exception):
    def __init__(self, order_id):
        msg = (f"TP/SL order with id {order_id} was not found. "
               f"It has been removed or never submitted")
        super().__init__(msg)


class PositionEntry:
    """PositionEntry represents amount of assets available for trading
    and gained as a result of order execution."""

    def __init__(self, amount, fill_price):
        self._tp_amount = 0
        self._sl_amount = 0
        self._acquired = 0
        self._amount = amount
        self._fill_price = fill_price
        self._tp = OrderedDict()
        self._sl = OrderedDict()

    @property
    def amount(self):
        return self._amount

    @property
    def acquired(self):
        return self._acquired

    @property
    def fill_price(self):
        return self._fill_price

    def acquire(self, amount):
        if self._acquired + amount > self._amount:
            raise NotEnoughInPosition(amount, self.amount - self.acquired)
        self._acquired += amount

    def release(self, amount):
        if amount > self._acquired:
            raise ReleaseMoreThanAcquired(amount, self._acquired)
        self._acquired -= amount

    def open_tp(self, order_id, amount):
        self._tp_amount += amount
        to_acquire = max(0, self._tp_amount - self._sl_amount)
        if self._acquired + to_acquire > self._amount:
            raise NotEnoughInPosition(to_acquire, self.amount - self.acquired)

        self._acquired += to_acquire
        self._tp[order_id] = amount
        return to_acquire

    def open_sl(self, order_id, amount):
        self._sl_amount += amount
        to_acquire = max(0, self._sl_amount - self._tp_amount)
        if self._acquired + to_acquire > self._amount:
            raise NotEnoughInPosition(to_acquire, self.amount - self.acquired)

        self._acquired += to_acquire
        self._sl[order_id] = amount
        return to_acquire

    def release_tp(self, order_id):
        try:
            amount = self._tp.pop(order_id)
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            before = self.acquired
            self._tp_amount -= amount
            self._acquired = min(before, max(self._tp_amount, self._sl_amount))
            to_release = before - self._acquired
            return to_release

    def release_sl(self, order_id):
        try:
            amount = self._sl.pop(order_id)
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            before = self.acquired
            self._sl_amount -= amount
            self._acquired = min(before, max(self._tp_amount, self._sl_amount))
            to_release = before - self._acquired
            return to_release

    def close(self, amount):
        # if amount > (self._amount - self.acquired):
        if amount > self._amount:
            raise NotEnoughInPosition(amount, self._amount)
        self._amount -= amount
        self._acquired -= amount

    def close_tp(self, order_id):
        try:
            amount = self._tp.pop(order_id)
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            self._tp_amount -= amount
            self._acquired -= amount
            self._amount -= amount

            to_release = 0
            cancelled = []
            sl_iter = iter(list(self._sl.keys()))  # iter over a copy of keys

            while self._sl_amount > self._acquired:
                sl_id = next(sl_iter) # stop iteration?
                to_release += self.release_sl(sl_id)
                cancelled.append(sl_id)
            return amount, to_release, cancelled

    def close_sl(self, order_id):
        try:
            amount = self._sl.pop(order_id)
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            self._sl_amount -= amount
            self._acquired -= amount
            self._amount -= amount

            to_release = 0
            cancelled = []
            tp_iter = iter(list(self._tp.keys()))  # iter over a copy of keys

            while self._tp_amount > self._acquired:
                tp_id = next(tp_iter) # stop iteration?
                to_release += self.release_tp(tp_id)
                cancelled.append(tp_id)
            return amount, to_release, cancelled


@dataclass
class EntryData:
    amount: Decimal
    fill_price: Decimal


class Position:
    """Position represents amount of assets available for trading
    and gained as a result of orders execution.

    It stores a sequence of PositionEntry objects. 
    They are the entry points of the position, so the position can contain
    multiple entries, each of them with different fill price and amount.

    Those entries are removed on closing the position, either by TP/SL
    or by an ordinary order. Both the position, and the position entry
    can be closed partially. In this case, the amount is decreased,
    but the entries would not be removed until the amount in entry = 0.
    TP/SL would not be cancelled either, while there are enough assets
    in the entry to make the order execution possible."""

    '''
    Some requirements to entries:
        - should be fast at access with index
        - must be possible to retrieve the last item
        - should be fast at deleting an item from mid, begin and end
        - must be possible to iterate over with guaranteed order (for close())
    '''
    def __init__(self):
        self._ids_counter = count()
        self._entries = OrderedDict()
        self._orders_to_entries = dict()

    def __len__(self):
        return len(self._entries)

    @property
    def total_amount(self):
        return sum(map(lambda x: x.amount, self._entries.values()))

    @property
    def available_amount(self):
        return sum(map(lambda x: x.amount - x.acquired, self._entries.values()))

    @property
    def entries(self):
        return [ 
            EntryData(amount=x.amount, fill_price=x.fill_price) 
                for x in self._entries.values() 
        ]

    def open(self, amount, fill_price):
        pos_id = next(self._ids_counter)
        self._entries[pos_id] = PositionEntry(amount, fill_price)

    def acquire(self, amount):
        iter_id = iter(self._entries.keys())
        available = self.available_amount  # for exception

        while amount:
            try:
                entry_id = next(iter_id)
                entry = self._entries[entry_id]
            except StopIteration as e:
                raise NotEnoughInPosition(amount, available)
            else:
                to_acquire = min(amount, entry.amount - entry.acquired) # min(amount, available)
                entry.acquire(to_acquire)
                amount -= to_acquire

    def release(self, amount):
        iter_id = iter(self._entries.keys())
        acquired = self.total_amount - self.available_amount  # for exception

        while amount:
            try:
                entry_id = next(iter_id)
                entry = self._entries[entry_id]
            except StopIteration as e:
                raise ReleaseMoreThanAcquired(amount, acquired)
            else:
                to_release = min(amount, entry.acquired) # min(amount, available)
                entry.release(to_release)
                amount -= to_release

    def close(self, amount) -> t.List[EntryData]:
        result = []
        # Iterate over a copy of entries keys to delete entries safely
        iter_id = iter(list(self._entries.keys()))
        total_amount = self.total_amount  # for exception

        while amount:
            try:
                # entry_id, entry = next(pos_iter)
                entry_id = next(iter_id)
                entry = self._entries[entry_id]  # Possible key error?
            except StopIteration as e:
                raise NotEnoughInPosition(amount, total_amount)
            else:
                to_remove = min(amount, entry.amount)
                amount -= to_remove
                entry.close(to_remove)
                if not entry.amount:
                    del self._entries[entry_id]
                result.append(EntryData(to_remove, entry.fill_price))
        return result

    def open_tp(self, order_id, amount):
        try:
            pos_id, entry = next(reversed(self._entries.items()))
            #print(entry.amount, entry.acquired)
        except StopIteration as e:
            raise EmptyPosition() from e
        else:
            to_acquire = entry.open_tp(order_id, amount)
            self._orders_to_entries[order_id] = pos_id
            return to_acquire

    def open_sl(self, order_id, amount):
        try:
            pos_id, entry = next(reversed(self._entries.items()))
            #print(entry.amount, entry.acquired)
        except StopIteration as e:
            raise EmptyPosition() from e
        else:
            to_acquire = entry.open_sl(order_id, amount)
            self._orders_to_entries[order_id] = pos_id
            return to_acquire

    def release_tp(self, order_id):
        try:
            position_id = self._orders_to_entries.pop(order_id)
            entry = self._entries[position_id]
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            to_release = entry.release_tp(order_id)
            if not entry.amount:
                del self._entries[position_id]
            return to_release

    def release_sl(self, order_id):
        try:
            position_id = self._orders_to_entries.pop(order_id)
            entry = self._entries[position_id]
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            to_release = entry.release_sl(order_id)
            if not entry.amount:
                del self._entries[position_id]
            return to_release

    def close_tp(self, order_id):
        try:
            position_id = self._orders_to_entries.pop(order_id)
            entry = self._entries[position_id]
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            amount, to_release, cancelled = entry.close_tp(order_id)
            if not entry.amount:
                del self._entries[position_id]
            for order_id in cancelled:
                del self._orders_to_entries[order_id]
            return EntryData(amount, entry.fill_price), to_release, cancelled

    def close_sl(self, order_id):
        try:
            position_id = self._orders_to_entries.pop(order_id)
            entry = self._entries[position_id]  #
        except KeyError as e:
            raise TpSlNotFound(order_id) from e
        else:
            amount, to_release, cancelled = entry.close_sl(order_id)
            if not entry.amount:
                del self._entries[position_id]
            for order_id in cancelled:
                del self._orders_to_entries[order_id]
            return EntryData(amount, entry.fill_price), to_release, cancelled

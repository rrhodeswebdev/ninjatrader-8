from decimal import Decimal
from backintime.broker.futures.position import Position


def test_open_tpsl_single_entry():
    pos = Position()
    pos.open(2, fill_price=100)

    tp_id = 0
    sl_id = 1

    to_acquire = pos.open_tp(tp_id, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id, 2)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].amount == 2
    assert pos._entries[0].acquired == 2


def test_release_tpsl_single_entry():
    pos = Position()
    pos.open(2, fill_price=100)

    tp_id = 0
    sl_id = 1

    to_acquire = pos.open_tp(tp_id, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id, 2)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].amount == 2
    assert pos._entries[0].acquired == 2

    to_release = 0

    to_release += pos.release_tp(tp_id)
    assert to_release == 0

    to_release += pos.release_sl(sl_id)
    assert to_release == 2
    assert len(pos) == 1


def test_close_tp_single_entry():
    pos = Position()
    pos.open(2, fill_price=100)

    tp_id = 0
    sl_id = 1

    to_acquire = pos.open_tp(tp_id, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id, 2)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].amount == 2
    assert pos._entries[0].acquired == 2

    entry_data, to_release, cancelled = pos.close_tp(tp_id)
    assert to_release == 0
    assert len(pos) == 0
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100


def test_close_sl_single_entry():
    pos = Position()
    pos.open(2, fill_price=100)

    tp_id = 0
    sl_id = 1

    to_acquire = pos.open_tp(tp_id, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id, 2)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].amount == 2
    assert pos._entries[0].acquired == 2

    entry_data, to_release, cancelled = pos.close_sl(sl_id)
    assert to_release == 0
    assert len(pos) == 0
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100


def test_open_many_tpsl_single_entry():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6


def test_release_many_tpsl_single_entry():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6

    to_release = 0
    to_release += pos.release_tp(tp_id_1)
    to_release += pos.release_sl(sl_id_1)
    to_release += pos.release_tp(tp_id_2)
    to_release += pos.release_sl(sl_id_2)

    assert to_release == 6
    assert len(pos) == 1


def test_close_many_tp_single_entry():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6

    to_release = 0
    entry_data, to_release, cancelled = pos.close_tp(tp_id_1)
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100

    entry_data, to_release, cancelled = pos.close_tp(tp_id_2)
    assert entry_data.amount == 4
    assert entry_data.fill_price == 100

    assert to_release == 0
    assert len(pos) == 0


def test_close_many_sl_single_entry():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6

    to_release = 0
    entry_data, to_release, cancelled = pos.close_sl(sl_id_1)
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100

    entry_data, to_release, cancelled = pos.close_sl(sl_id_2)
    assert entry_data.amount == 4
    assert entry_data.fill_price == 100

    assert to_release == 0
    assert len(pos) == 0


def test_close_tp_and_sl_single_entry():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6

    to_release = 0
    entry_data, to_release, cancelled = pos.close_tp(tp_id_1)
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100

    entry_data, to_release, cancelled = pos.close_sl(sl_id_2)
    assert entry_data.amount == 4
    assert entry_data.fill_price == 100

    assert to_release == 0
    assert len(pos) == 0


def test_close_tp_and_release():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6

    to_release = 0
    entry_data, to_release, cancelled = pos.close_tp(tp_id_1)
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100
    print(pos._entries[0].amount)
    print(pos._entries[0]._acquired)

    to_release += pos.release_tp(tp_id_2)
    to_release += pos.release_sl(sl_id_2)
    assert to_release == 4
    assert len(pos) == 1


def test_close_sl_and_release():
    pos = Position()
    pos.open(amount=6, fill_price=100)

    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3

    to_acquire = pos.open_tp(tp_id_1, 2)
    
    
    assert to_acquire == 2

    to_acquire = pos.open_sl(sl_id_1, 2)
    
    
    assert to_acquire == 0

    to_acquire= pos.open_tp(tp_id_2, 4)
    
    
    assert to_acquire == 4

    to_acquire= pos.open_sl(sl_id_2, 4)
    
    
    assert to_acquire == 0

    assert len(pos) == 1
    assert pos._entries[0].acquired == 6
    assert pos._entries[0].amount == 6

    to_release = 0
    entry_data, to_release, cancelled = pos.close_sl(sl_id_1)
    assert entry_data.amount == 2
    assert entry_data.fill_price == 100

    to_release += pos.release_tp(tp_id_2)
    to_release += pos.release_sl(sl_id_2)
    assert to_release == 4
    assert len(pos) == 1

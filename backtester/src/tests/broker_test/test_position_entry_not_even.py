from backintime.broker.futures.position import PositionEntry


def test_open_tpsl():
    tp_id = 0
    sl_id = 1

    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id, 6)
    
    assert to_acquire == 4
    assert entry.acquired == 6


def test_release_tpsl():
    tp_id = 0
    sl_id = 1
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id, 6)
    
    assert to_acquire == 4
    assert entry.acquired == 6

    to_release = 0
    to_release += entry.release_tp(tp_id)
    assert to_release == 0
    to_release += entry.release_sl(sl_id)
    assert to_release == 6
    assert entry.acquired == 0
    assert entry.amount == 6


def test_close_tp():
    tp_id = 0
    sl_id = 1
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id, 6)
    
    assert to_acquire == 4
    assert entry.acquired == 6

    amount, to_release, cancelled = entry.close_tp(tp_id)
    assert to_release == 4
    assert entry.acquired == 0
    assert entry.amount == 4


def test_close_sl():
    tp_id = 0
    sl_id = 1
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id, 6)
    
    assert to_acquire == 4
    assert entry.acquired == 6

    amount, to_release, cancelled = entry.close_sl(sl_id)
    assert to_release == 0
    assert entry.acquired == 0
    assert entry.amount == 0


def test_open_many_tpsl():
    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3
    tp_id_3 = 4
 
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id_1, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id_1, 3)
    
    assert to_acquire == 1
    assert entry.acquired == 3

    to_acquire = entry.open_tp(tp_id_2, 2)
    
    assert to_acquire == 1
    assert entry.acquired == 4

    to_acquire = entry.open_sl(sl_id_2, 3)
    
    assert to_acquire == 2
    assert entry.acquired == 6

    to_acquire = entry.open_tp(tp_id_3, 2)
    
    assert to_acquire == 0
    assert entry.acquired == 6


def test_release_many_tpsl():
    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3
    tp_id_3 = 4
 
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id_1, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id_1, 3)
    
    assert to_acquire == 1
    assert entry.acquired == 3

    to_acquire = entry.open_tp(tp_id_2, 2)
    
    assert to_acquire == 1
    assert entry.acquired == 4

    to_acquire = entry.open_sl(sl_id_2, 3)
    
    assert to_acquire == 2
    assert entry.acquired == 6

    to_acquire = entry.open_tp(tp_id_3, 2)
    
    assert to_acquire == 0
    assert entry.acquired == 6

    to_release = 0
    to_release += entry.release_tp(tp_id_1)
    to_release += entry.release_sl(sl_id_1)
    to_release += entry.release_tp(tp_id_2)
    to_release += entry.release_sl(sl_id_2)
    to_release += entry.release_tp(tp_id_3)
    assert to_release == 6
    assert entry.acquired == 0
    assert entry.amount == 6


def test_close_many_tp():
    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3
    tp_id_3 = 4
 
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id_1, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id_1, 3)
    
    assert to_acquire == 1
    assert entry.acquired == 3

    to_acquire = entry.open_tp(tp_id_2, 2)
    
    assert to_acquire == 1
    assert entry.acquired == 4

    to_acquire = entry.open_sl(sl_id_2, 3)
    
    assert to_acquire == 2
    assert entry.acquired == 6

    to_acquire = entry.open_tp(tp_id_3, 2)
    
    assert to_acquire == 0
    assert entry.acquired == 6

    amount, to_release, cancelled = entry.close_tp(tp_id_1)
    assert to_release == 0

    amount, to_release, cancelled = entry.close_tp(tp_id_2)
    assert to_release == 0

    amount, to_release, cancelled = entry.close_tp(tp_id_3)
    assert to_release == 0
    assert entry.acquired == 0
    assert entry.amount == 0


def test_close_many_sl():
    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3
    tp_id_3 = 4
 
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id_1, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id_1, 3)
    
    assert to_acquire == 1
    assert entry.acquired == 3

    to_acquire = entry.open_tp(tp_id_2, 2)
    
    assert to_acquire == 1
    assert entry.acquired == 4

    to_acquire = entry.open_sl(sl_id_2, 3)
    
    assert to_acquire == 2
    assert entry.acquired == 6

    to_acquire = entry.open_tp(tp_id_3, 2)
    
    assert to_acquire == 0
    assert entry.acquired == 6

    amount, to_release, cancelled = entry.close_sl(sl_id_1)
    assert to_release == 0

    amount, to_release, cancelled = entry.close_sl(sl_id_2)
    assert to_release == 0
    assert entry.acquired == 0
    assert entry.amount == 0


def test_close_tp_and_release():
    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3
    tp_id_3 = 4
 
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id_1, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id_1, 3)
    
    assert to_acquire == 1
    assert entry.acquired == 3

    to_acquire = entry.open_tp(tp_id_2, 2)
    
    assert to_acquire == 1
    assert entry.acquired == 4

    to_acquire = entry.open_sl(sl_id_2, 3)
    
    assert to_acquire == 2
    assert entry.acquired == 6

    to_acquire = entry.open_tp(tp_id_3, 2)
    
    assert to_acquire == 0
    assert entry.acquired == 6

    amount, to_release, cancelled = entry.close_tp(tp_id_1)
    assert to_release == 0

    to_release += entry.release_tp(tp_id_2)
    to_release += entry.release_sl(sl_id_2)
    to_release += entry.release_tp(tp_id_3)
    assert to_release == 4
    assert entry.acquired == 0
    assert entry.amount == 4


def test_close_sl_and_release():
    tp_id_1 = 0
    sl_id_1 = 1
    tp_id_2 = 2
    sl_id_2 = 3
    tp_id_3 = 4
 
    entry = PositionEntry(amount=6, fill_price=100)
    to_acquire = entry.open_tp(tp_id_1, 2)
    
    assert to_acquire == 2
    assert entry.acquired == 2

    to_acquire = entry.open_sl(sl_id_1, 3)
    
    assert to_acquire == 1
    assert entry.acquired == 3

    to_acquire = entry.open_tp(tp_id_2, 2)
    
    assert to_acquire == 1
    assert entry.acquired == 4

    to_acquire = entry.open_sl(sl_id_2, 3)
    
    assert to_acquire == 2
    assert entry.acquired == 6

    to_acquire = entry.open_tp(tp_id_3, 2)
    
    assert to_acquire == 0
    assert entry.acquired == 6

    amount, to_release, cancelled = entry.close_sl(sl_id_1)
    assert to_release == 0

    to_release += entry.release_sl(sl_id_2)
    to_release += entry.release_tp(tp_id_3)
    assert to_release == 3
    assert entry.acquired == 0
    assert entry.amount == 3
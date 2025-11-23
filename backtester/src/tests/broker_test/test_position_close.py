from backintime.broker.futures.position import Position


def test_open():
    pos = Position()
    pos.open(6, fill_price=100)
    assert len(pos) == 1


def test_close():
    pos = Position()
    pos.open(6, fill_price=100)
    assert len(pos) == 1

    entries = pos.close(6)
    assert len(pos) == 0
    assert len(entries) == 1
    assert entries[0].amount == 6
    assert entries[0].fill_price == 100


def test_close_partially():
    pos = Position()
    pos.open(6, fill_price=100)
    assert len(pos) == 1

    entries = pos.close(3)
    assert len(pos) == 1
    assert len(entries) == 1
    assert entries[0].amount == 3
    assert entries[0].fill_price == 100


def test_open_many():
    pos = Position()
    pos.open(2, fill_price=100)
    pos.open(4, fill_price=110)
    assert len(pos) == 2


def test_close_many():
    pos = Position()
    pos.open(2, fill_price=100)
    pos.open(4, fill_price=110)
    assert len(pos) == 2

    entries = pos.close(6)
    assert len(pos) == 0
    assert len(entries) == 2
    assert entries[0].amount == 2
    assert entries[0].fill_price == 100
    assert entries[1].amount == 4
    assert entries[1].fill_price == 110


def test_close_many_partially():
    pos = Position()
    pos.open(2, fill_price=100)
    pos.open(4, fill_price=110)
    assert len(pos) == 2

    entries = pos.close(5)
    assert len(pos) == 1
    assert len(entries) == 2
    assert entries[0].amount == 2
    assert entries[0].fill_price == 100
    assert entries[1].amount == 3
    assert entries[1].fill_price == 110
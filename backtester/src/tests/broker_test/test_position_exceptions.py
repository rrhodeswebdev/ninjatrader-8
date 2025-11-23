from decimal import Decimal

import pytest

from backintime.broker.futures.position import (Position, 
                            EmptyPosition, NotEnoughInPosition, 
                            ReleaseMoreThanAcquired, TpSlNotFound)


@pytest.fixture
def position():
    pos = Position()
    pos.open(2, fill_price=100)
    return pos


@pytest.fixture
def empty_position():
    return Position()


def test_acquire_too_much(position):
    exception_raised = False
    try:
        position.acquire(3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_release_more_than_acquired(position):
    exception_raised = False
    try:
        position.release(1)
    except ReleaseMoreThanAcquired as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_close_too_much(position):
    exception_raised = False
    try:
        position.close(3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_open_tp_with_too_large_amount(position):
    exception_raised = False
    try:
        position.open_tp(order_id=0, amount=3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_open_sl_with_too_large_amount(position):
    exception_raised = False
    try:
        position.open_sl(order_id=0, amount=3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_open_tp_empty_position(empty_position):
    exception_raised = False
    try:
        empty_position.open_tp(order_id=0, amount=3)
    except EmptyPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_open_sl_empty_position(empty_position):
    exception_raised = False
    try:
        empty_position.open_sl(order_id=0, amount=3)
    except EmptyPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_release_non_existent_tp(position):
    exception_raised = False
    try:
        position.release_tp(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_release_non_existent_sl(position):
    exception_raised = False
    try:
        position.release_sl(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_close_non_existent_tp(position):
    exception_raised = False
    try:
        position.close_tp(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_close_non_existent_sl(position):
    exception_raised = False
    try:
        position.close_sl(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised

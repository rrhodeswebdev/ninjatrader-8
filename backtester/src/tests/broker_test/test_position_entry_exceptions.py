from decimal import Decimal

import pytest

from backintime.broker.futures.position import (PositionEntry, 
                NotEnoughInPosition, ReleaseMoreThanAcquired, TpSlNotFound)


@pytest.fixture
def entry():
    return PositionEntry(amount=2, fill_price=100)


def test_acquire_too_much(entry):
    exception_raised = False
    try:
        entry.acquire(3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_release_more_than_acquired(entry):
    exception_raised = False
    try:
        entry.release(1)
    except ReleaseMoreThanAcquired as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_close_too_much(entry):
    exception_raised = False
    try:
        entry.close(3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_open_tp_with_too_large_amount(entry):
    exception_raised = False
    try:
        entry.open_tp(order_id=0, amount=3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_open_sl_with_too_large_amount(entry):
    exception_raised = False
    try:
        entry.open_sl(order_id=0, amount=3)
    except NotEnoughInPosition as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_release_non_existent_tp(entry):
    exception_raised = False
    try:
        entry.release_tp(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_release_non_existent_sl(entry):
    exception_raised = False
    try:
        entry.release_sl(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_close_non_existent_tp(entry):
    exception_raised = False
    try:
        entry.close_tp(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised


def test_close_non_existent_sl(entry):
    exception_raised = False
    try:
        entry.close_sl(order_id=0)
    except TpSlNotFound as e:
        print(e)
        exception_raised = True
    assert exception_raised

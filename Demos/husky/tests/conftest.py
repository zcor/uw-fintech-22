#!/usr/bin/python3

import pytest


@pytest.fixture(scope="function", autouse=True)
def isolate(fn_isolation):
    # perform a chain rewind after completing each test, to ensure proper isolation
    # https://eth-brownie.readthedocs.io/en/v1.10.3/tests-pytest-intro.html#isolation-fixtures
    pass


@pytest.fixture(scope="module")
def token(Husky, accounts):
    h = Husky.deploy({'from': accounts[0]})
    h.mint(accounts[0], 10 ** 21, {'from': accounts[0]})
    return h

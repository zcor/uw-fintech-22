#!/usr/bin/python3

from brownie import HuskyTokenDeployer, accounts, HuskyToken, HuskyTokenMinter, Wei
def main():
    provost = accounts.load('husky')
    token = HuskyToken.deploy("Husky Inu", "HUSKYINU", 0, {'from': provost}, publish_source=True)
    minter = HuskyTokenMinter.deploy(token, provost, {'from': provost}, publish_source=True)
    token.addMinter(minter, {'from': provost});
    token.renounceMinter({'from': provost});


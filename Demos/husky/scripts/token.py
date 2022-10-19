#!/usr/bin/python3

from brownie import Husky, accounts, network


def main():
    if network.show_active() == 'goerli':
        acct = accounts.load('husky')
    else:
        acct = accounts[0]

    mint_addrs = [
        '0x43050ff346a298008F17A063C55c9a1aea353b6F', # Majid
        '0xBa63B34cae23C9c85333FbcABF6c463DE2ad9eB3', # Tracy
        '0x8032547871D51848bf238740bc4E755445643Aa6', # Aarti
        '0xCfD2f937cf1f8cdC6d83bE8f65ac8a4Bf87C2216', # Peter
        '0x382f9430a07e6c3F478d2677ba15eE3e82a613BD', # Jon M
        '0xAF2F32ed62b20B1640F27C67B9C6733b83924a4E', # Jon K
        '0xEA2406e9EB5fFD3DBb73337fE9b187CF555e78cd', # Dan M
        '0x3892F58040Bc631d75398b320A3C617C371B205D', # Chandler 1
        '0x0cA9D248B4A535Ad62Dd9897e012502D0c2328AC', # Chandler 2
        ]

    husky = Husky.deploy({'from': acct})
    
    for addr in mint_addrs:
        husky.mint(addr, 1_000_000 * 10 ** 18 , {'from': acct})

    return husky

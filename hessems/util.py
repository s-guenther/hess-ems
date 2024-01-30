#!/usr/bin/env python3
"""This module contains various shared helper functions used by multiple
files."""

from copy import copy


def update_std(new, std):
    """Uses the dict `std` as a base and updates all keys that are present
    in `new` with the values of `new`. Returns the `std` dict, if `new is
    None`. Throws a ValueError, if `new` contains unknown keys."""
    # Return standard parameters if no updated parameters are provided
    if new is None:
        return std

    # Test if provided updated para names are valid/within the std para dict
    new_keys = set(new.keys())
    std_keys = set(std.keys())
    if not new_keys.issubset(std_keys):
        unkown_keys = new_keys.difference(std_keys)
        msg = (f'Unknown parameters: {unkown_keys}. The standard parameters '
               f'define: {std_keys}')
        raise ValueError(msg)

    # Update the std dict with the new values
    updated = copy(std)
    for key, val in new:
        updated[key] = val
    return updated


def subdict(superdict, keys):
    """Returns a new dict with `keys` keys, containing the vals from
    `superdict`. If any key in `keys` is not present in `superdict`,
    a KeyError will be thrown."""
    return {k: superdict[k] for k in keys}
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in transfer-engine interface and connectors.

Import concrete connectors from .connector; importing the transport contract
must not import the protocol runtime or native engines.
"""

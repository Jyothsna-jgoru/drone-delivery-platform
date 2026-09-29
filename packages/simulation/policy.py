from __future__ import annotations

import numpy as np

from packages.shared.models import Action


class RuleBasedPolicy:
    """Safe pipeline validation policy; useful before and alongside learned policies."""

    def act(self, env, agent: str) -> Action:
        drone = env.drones[agent]
        package = env.packages[drone["assigned_package"]]
        mask = env.action_mask(agent)
        if not package.get("active", True) or package["delivered"]:
            return Action.HOVER
        if drone["battery"] < 0.2:
            for action in (Action.CHARGE, Action.MOVE_TO_CHARGER, Action.RETURN_TO_BASE):
                if mask[action]:
                    return action
        if mask[Action.PICK_UP]:
            return Action.PICK_UP
        if mask[Action.DELIVER]:
            return Action.DELIVER
        target = package["destination"] if package["picked_up"] else package["pickup"]
        delta = target - drone["position"]
        candidates = []
        if abs(delta[2]) > 0.5:
            candidates.append(Action.CLIMB if delta[2] > 0 else Action.DESCEND)
        # Route around the configured central geofence through its southern corridor.
        if target[0] > 12 and drone["position"][0] < 13:
            if drone["position"][1] >= 8:
                candidates.append(Action.MOVE_SOUTH)
            else:
                candidates.append(Action.MOVE_EAST)
        elif delta[1] > 0.5 and not mask[Action.MOVE_NORTH]:
            candidates.append(Action.MOVE_EAST)
        elif abs(delta[0]) >= abs(delta[1]):
            candidates.extend([Action.MOVE_EAST if delta[0] > 0 else Action.MOVE_WEST, Action.MOVE_NORTH if delta[1] > 0 else Action.MOVE_SOUTH])
        else:
            candidates.extend([Action.MOVE_NORTH if delta[1] > 0 else Action.MOVE_SOUTH, Action.MOVE_EAST if delta[0] > 0 else Action.MOVE_WEST])
        for action in candidates:
            if mask[action]:
                proposed = drone["position"] + env.safety.ACTION_VELOCITY.get(action, np.zeros(3))
                if not env.safety._in_restricted_zone(proposed):
                    return action
        return Action.HOVER

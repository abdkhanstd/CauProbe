"""Commissioning text: what a gateway is told about its building, in prose.

Real commissioning knowledge arrives as datasheets, BMS point lists, network
diagrams pasted into tickets, and handover notes written by people. This
module generates that text for each plant, from cue phrases tied to each
mechanism factor, with the properties that make reading it hard:

* coverage varies: a well-documented site mentions most factors, a poorly
  documented one few (the expert should then be uncertain, not guess);
* some cues are indirect and need domain knowledge ("LoRaWAN Class A end
  nodes" means a sleeping MAC without saying so);
* some notes are stale or wrong (the original design, not what was built);
* distractor sentences mention irrelevant equipment.

Cue phrases and sentence frames are split into **disjoint train and test
vocabularies**. The expert is trained on the first and evaluated on the second,
so held-out accuracy measures whether it knows what a device *is*, not whether
it memorised a phrase. A bag-of-words classifier cannot cross that split; a
pretrained language model might.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# factor -> value -> (train cues, test cues)
CUES: Dict[str, Dict[int, Tuple[List[str], List[str]]]] = {
    "A": {
        0: (["802.11ac access points with frame aggregation",
             "always-on Wi-Fi radios in the sensor fleet",
             "Wi-Fi 6 clients using A-MPDU batching",
             "the sensors join the corporate WLAN and batch frames when duty-cycled"],
            ["dual-band 802.11n access points; sensors aggregate uplink frames",
             "mains-powered Wi-Fi sensor nodes on a managed WLAN",
             "an 802.11ax network where nodes buffer and burst their readings",
             "Ethernet-backed Wi-Fi mesh, radios stay associated continuously"]),
        1: (["Zigbee sleepy end devices",
             "ContikiMAC radio duty cycling on the motes",
             "a low-power-listening MAC that sleeps between checks",
             "battery sensors that wake periodically to poll their parent"],
            ["Thread sleepy end devices (SEDs) polling a border router",
             "LoRaWAN Class A end nodes",
             "BLE peripherals with long connection intervals",
             "802.15.4 nodes running a receiver-initiated wake-up MAC"]),
    },
    "B": {
        0: (["fibre to the building and a cloud region in the same city",
             "the gateway is a Raspberry Pi 3 whose CPU sits near 85% running Node-RED",
             "local analytics run on a single-core ARM gateway that saturates at peak",
             "gigabit metro Ethernet uplink"],
            ["a 10 GbE campus backbone with the analytics VM two hops away",
             "an Intel Atom gateway that is CPU-bound once the floor is busy",
             "the edge box is underpowered; local inference queues up at lunchtime",
             "a low-latency enterprise WAN to a nearby data centre"]),
        1: (["the only uplink is an LTE Cat-M1 modem",
             "a rural 4G router provides backhaul",
             "the cloud tenant is hosted on another continent over VPN",
             "satellite backhaul with high round-trip time"],
            ["Starlink is the site's only internet connection",
             "a VSAT terminal on the roof carries all upstream traffic",
             "NB-IoT backhaul through the operator's core network",
             "the analytics backend sits in a distant region behind a congested MPLS link"]),
    },
    "C": {
        0: (["a dedicated Mosquitto broker serves only this floor",
             "single-tenant private MQTT broker on the gateway",
             "about 40 sensors publish to our own broker",
             "no other systems share the broker"],
            ["an EMQX instance reserved for this deployment alone",
             "the broker is ours; the landlord's systems have their own",
             "four dozen environmental sensors report to the local broker",
             "MQTT traffic is exclusively from the floor's own devices"]),
        1: (["the MQTT broker is shared with twelve other tenants",
             "the landlord's BMS also publishes to this broker",
             "multi-tenant HiveMQ cluster operated by the building owner",
             "only half a dozen of our sensors use the shared broker"],
            ["a shared AWS IoT Core endpoint used by every tenant in the tower",
             "co-working operators, lifts and access control all publish to the same broker",
             "our eight devices are a small fraction of the broker's subscribers",
             "the building's central RabbitMQ MQTT plugin is shared across floors"]),
    },
    "D": {
        0: (["post-tensioned concrete slab with exposed soffits",
             "1920s solid brick masonry with heavy floors",
             "thermally activated building structures (TABS) in the slab",
             "high thermal mass construction"],
            ["a rammed-earth and stone heritage building",
             "precast hollow-core concrete planks left exposed",
             "thick limestone walls and a screeded concrete deck",
             "an exposed-concrete Passivhaus-style structure"]),
        1: (["timber frame with lightweight partitions",
             "a modular portable office cabin",
             "raised floor and suspended ceiling throughout, gypsum partitions",
             "glass curtain wall over a light steel frame"],
            ["cross-laminated timber floors with a carpeted raised access floor",
             "a prefabricated container-based office",
             "drywall partitions on metal studs and a lightweight roof deck",
             "an inflatable or tensile-membrane structure"]),
    },
}

FRAMES = (
    ["Site survey notes: {}.", "Per the handover document, {}.",
     "Integrator remark: {}.", "Network diagram annotation: {}.",
     "From the asset register: {}."],
    ["Commissioning log entry: {}.", "As-built drawings indicate {}.",
     "Field engineer wrote: {}.", "Ticket comment from facilities: {}.",
     "Vendor datasheet summary: {}."],
)
STALE = (
    ["Original tender specified {}, which may have changed.",
     "An old revision of the drawings shows {}."],
    ["The pre-refurbishment survey listed {}.",
     "Design intent (not verified on site) was {}."],
)
DISTRACT = [
    "The lighting uses DALI ballasts on a separate bus.",
    "Fire alarm panels report over a dedicated loop.",
    "Lift controllers are out of scope for this gateway.",
    "Badge readers were upgraded in the last quarter.",
    "Water meters are read monthly by the utility.",
    "The kitchen has its own extract fan schedule.",
    "Parking sensors belong to the car-park operator.",
    "Signage displays are managed by marketing.",
]

QUESTIONS = [
    ("A", "radio_sleeps"),        # yes -> A1
    ("B", "wan_bottleneck"),      # yes -> B1
    ("C", "shared_broker"),       # yes -> C1
    ("D", "lightweight_building"),  # yes -> D1
]

PROMPT_HEAD = ("You commission IoT gateways. Read the site notes and answer each "
               "question with yes or no.\n"
               "radio_sleeps: does the sensor radio sleep between transmissions, "
               "so a lower duty cycle adds delay?\n"
               "wan_bottleneck: is the uplink to the cloud slow, so offloading adds "
               "latency?\n"
               "shared_broker: is the MQTT broker dominated by other tenants' traffic?\n"
               "lightweight_building: is the building of light construction with "
               "little thermal mass?\n")


def make_notes(combo, rng, split: str = "train", quality: float = None,
               p_stale: float = 0.08) -> Tuple[str, Dict[str, int]]:
    """Commissioning text for one plant, and which factors it actually covers.

    `quality` is the per-factor probability of a correct mention; if None it
    is drawn per plant from {0.25, 0.6, 0.9} (poor / average / good records).
    """
    si = 0 if split == "train" else 1
    if quality is None:
        quality = float(rng.choice([0.25, 0.6, 0.9]))
    truth = dict(zip("ABCD", combo))
    sents, covered = [], {}
    for F in "ABCD":
        v = truth[F]
        u = rng.random()
        if u < quality:
            cue = rng.choice(CUES[F][v][si])
            sents.append(rng.choice(FRAMES[si]).format(cue))
            covered[F] = 1
        elif u < quality + p_stale:
            cue = rng.choice(CUES[F][1 - v][si])
            sents.append(rng.choice(STALE[si]).format(cue))
            covered[F] = -1
        else:
            covered[F] = 0
    for _ in range(int(rng.integers(1, 4))):
        sents.append(str(rng.choice(DISTRACT)))
    order = rng.permutation(len(sents))
    text = " ".join(sents[i] for i in order)
    return text, covered


def make_ticket(factor: str, new_value: int, rng, split: str = "test") -> str:
    si = 0 if split == "train" else 1
    cue = rng.choice(CUES[factor][new_value][si])
    lead = rng.choice(["Maintenance ticket closed: the site now has",
                       "Change record: replaced equipment; we now run",
                       "Work order completed, the floor was migrated to"])
    return f"{lead} {cue}."


def prompt_for(text: str) -> str:
    return PROMPT_HEAD + "Site notes: " + text + "\nAnswers:\n"

"""
Every utility method related to Tonnetz graph storage and processing is stuffed in here.

No I will not document anything.
"""

from pathlib import Path
from typing import List

import hexy as hx
import matplotlib.pyplot as plt
import music21 as m21
import numpy as np
from PIL import Image, ImageDraw, ImageFont


axial_to_pixel_mat = np.array([
    [3/2, 0],
    [np.sqrt(3) / 2, np.sqrt(3)],
])


def sanitize_coordinates(coordinates):
    if not isinstance(coordinates, (np.ndarray, list, tuple)):
        raise TypeError("Coordinates must be a numpy array or list of axial coordinates")
    coordinates = np.array(coordinates)
    if len(coordinates.shape) == 1:
        coordinates = np.array([coordinates])
    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must be Nx2.")
    return coordinates


def axial2oddq(axial):
    axial = np.array(axial)
    return np.array([axial[..., 1] + (axial[..., 0] - (axial[..., 0] & 1)) // 2, axial[..., 0]]).T


def oddq2axial(oddq):
    oddq = np.array(oddq)
    return np.array([oddq[..., 1], oddq[..., 0] - (oddq[..., 1] - (oddq[..., 1] & 1)) // 2]).T


def axial_to_notes(axial):
    """3-4-5 Tonnetz mapping of axial coordinates to MIDI note value"""
    axial = np.array(axial)
    if len(axial.shape) == 1:
        if axial.shape[0] != 2:
            raise ValueError("axial must be a 2-element array")
        return np.sum([-3, -7] * axial) + 60
    if len(axial.shape) != 2:
        raise ValueError("axial must be an Nx2 array")
    if axial.shape[1] != 2:
        raise ValueError("axial must be an Nx2 array")
    return np.sum([-3, -7] * axial, axis=1) + 60


def get_ring(center, radius):
    """get_ring but axial instead"""
    center = np.array([center[0], -center[0] - center[1], center[1]])
    return hx.cube_to_axial(hx.get_ring(center, radius)).astype(int)


def center2pixel(centers, radius):
    centers = sanitize_coordinates(centers)
    pixelcoords = (axial_to_pixel_mat @ centers.T * radius).T
    if len(centers.shape):
        return pixelcoords[0]
    return pixelcoords


# generate finite Tonnetz coordinates
notenum2axial = {}
notenum2axial[60] = (0, 0)
notenum2oddq = {}
notenum2oddq[60] = (0, 0)
allnotesfound = False
radiussearch = 1
while not allnotesfound:
    ring = get_ring([0, 0], radiussearch)
    ringasnotes = axial_to_notes(ring)
    for i, note in enumerate(ringasnotes):
        if note not in notenum2axial.keys() and note >= 21 and note <= 108:
            coord = tuple(ring[i])
            notenum2axial[note] = coord
            notenum2oddq[note] = tuple(axial2oddq(coord))
    # 88 key piano
    allnotesfound = len(notenum2axial.keys()) == 88
    radiussearch += 1

note88oddq = np.array(list(notenum2oddq.values()))
note88height = np.max(note88oddq[:, 0]) - np.min(note88oddq[:, 0]) + 1
note88width = np.max(note88oddq[:, 1]) - np.min(note88oddq[:, 1]) + 1
ODDQ_MIN_I = np.min(note88oddq[:, 0])
ODDQ_MIN_J = np.min(note88oddq[:, 1])


def oddqtonnetz_fromnotes(midinotes):
    note88oddq = np.array(list(notenum2oddq.values()))
    note88height = np.max(note88oddq[:, 0]) - ODDQ_MIN_I + 1
    note88width = np.max(note88oddq[:, 1]) - ODDQ_MIN_J + 1
    note88grid = np.zeros((note88height, note88width), dtype=float)
    imin = np.min(note88oddq[:, 0])
    jmin = np.min(note88oddq[:, 1])
    for note in midinotes:
        if note not in notenum2oddq.keys():
            continue
        noteoddq = notenum2oddq[note]
        note88grid[noteoddq[0] - imin, noteoddq[1] - jmin] = 1.0
    return note88grid


class TonnetzTile:
    def __init__(self, coordinate=None):
        """
        :param coordinate: Axial coordinate of the tile
        """
        self.coordinate = coordinate
        if self.coordinate is None:
            self.coordinates = (0, 0)
        self.midi_note = axial_to_notes(self.coordinate)

    @classmethod
    def from_midi_note(cls, midi_note):
        if midi_note not in notenum2axial.keys():
            raise ValueError(f"midi note {midi_note} not found in Tonnetz.")
        return cls(notenum2axial[midi_note])
    
    def to_polygon(self, radius):
        angs = 60 * np.pi / 180 * np.arange(0, 6)
        center = center2pixel(self.coordinate, radius)
        pts = np.array([center[0] + radius * np.cos(angs), center[1] + radius * np.sin(angs)]).T
        return pts
    
    def __repr__(self):
        return f"TonnetzTile({self.coordinate}, {self.midi_note})"
    
    def __str__(self):
        return f"TonnetzTile at {self.coordinate} with MIDI note {self.midi_note}"


class TonnetzMap:
    def __init__(self):
        self._map = {}

    def keys(self):
        yield from self._map.keys()

    def values(self):
        yield from self._map.values()

    def items(self):
        yield from self._map.items()

    def __len__(self):
        return self._map.__len__()

    def __iter__(self):
        yield from self._map

    def __contains__(self, key):
        key = tuple(key)
        return key in self._map

    def __setitem__(self, coordinates, hex_objects):
        """
        Assigns hex objects as values to coordinates as keys. The number of coordinates and hex objects
        should be equal.
        :param coordinates: Axial coordinates of each hex tile.
        :param hex_objects: the hex objects themselves.
        :return: None
        """
        coordinates = sanitize_coordinates(coordinates)
        if not isinstance(hex_objects, (np.ndarray, list, tuple)):
            hex_objects = [hex_objects]
        if len(hex_objects) == 1:
            # if length 1, just assume a broadcast
            hex_objects = hex_objects * len(coordinates)
        if len(coordinates) != len(hex_objects):
            raise ValueError("Number of coordinates does not match number of hex objects.")

        keys = list(map(tuple, coordinates))
        for key, hex in zip(keys, hex_objects):
            if key in self._map.keys():
                raise ValueError(f"key {key} already exists.")
            self._map[key] = hex

    def set_active(self, coordinates):
        coordinates = sanitize_coordinates(coordinates)
        hex_objects = [TonnetzTile(coord) for coord in coordinates]
        self.__setitem__(coordinates, hex_objects)

    def set_active_midi(self, midi_notes):
        midi_notes = np.array(midi_notes)
        if len(midi_notes.shape) == 0:
            midi_notes = np.array([midi_notes])
        if len(midi_notes.shape) != 1:
            raise ValueError("midi_notes must be a 1D array.")
        hex_objects = [TonnetzTile.from_midi_note(midi_note) for midi_note in midi_notes]
        self.__setitem__([hexobj.coordinate for hexobj in hex_objects], hex_objects)

    def setitem_direct(self, key, value):
        if key in self._map.keys():
            raise ValueError(f"key {key} already exists.")
        self._map[key] = value

    def overwrite_entries(self, coordinates, hex):
        keys = list(map(tuple, coordinates))
        for key in keys:
            self._map[key] = hex

    def to_oddq_grid(self, xlim=None, ylim=None):
        return oddqtonnetz_fromnotes([hex.midi_note for hex in self.values()])

    def __delitem__(self, coordinates):
        coordinates = sanitize_coordinates(coordinates)
        keys = list(map(tuple, coordinates))
        for key in keys:
            if key in self.keys():
                del self._map[key]

    def __getitem__(self, coordinates):
        """
        Retrieves hexes stores at `coordinates`
        :param coordinate: the locations used as keys for hexes. You can pass more than one coordinate
        :return: list of hexes mapped to using `coordinates`
        """
        coordinates = sanitize_coordinates(coordinates)
        keys = list(map(tuple, coordinates))
        return [self._map.get(k) for k in keys if k in self._map.keys()]
    
    def __repr__(self):
        return self._map.__repr__()
    
    def __str__(self):
        return self._map.__str__()
    
    def draw(self, center=None, width=800, height=800, radius=50, draw_unplayed=True):
        if center is None:
            center = np.array([0, 0])
        centerpix = np.array([width // 2, height // 2])
        img = Image.new("RGB", [width, height], 0)
        draw = ImageDraw.Draw(img, "RGB")

        centercoord = center2pixel(center, radius)
        offset = centerpix - centercoord
        # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        fnt = None
        draw.rectangle([(0,0), img.size], fill = (220, 220, 220))
        for coord, hex in self.items():
            hexcent = center2pixel(coord, radius) + offset
            hexvertices = hex.to_polygon(radius) + offset
            draw.polygon(list(map(tuple, hexvertices)), outline=(0, 0, 0, 255), fill=(255, 244, 38, 200))
            draw.text(tuple(hexcent), m21.note.Note(hex.midi_note).nameWithOctave, font=fnt, fill=(0, 0, 0, 255), align="center", anchor="mm")
        if draw_unplayed:
            for midival, coord in notenum2axial.items():
                if not self.__contains__(coord):
                    hexcent = center2pixel(coord, radius) + offset
                    hexvertices = TonnetzTile.from_midi_note(midival).to_polygon(radius) + offset
                    draw.polygon(list(map(tuple, hexvertices)), outline=(0, 0, 0, 255), fill=(200, 200, 200, 200))
                    draw.text(tuple(hexcent), m21.note.Note(midival).nameWithOctave, font=fnt, fill=(0, 0, 0, 255), align="center", anchor="mm")
        return img
    
    @classmethod
    def from_oddqgrid(cls, grid):
        """Assumes 13x9 finite Tonnetz grid, oddq hex coordinate format."""
        tm = cls()
        active_coords = []
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val:
                    active_coords.append(oddq2axial([i + ODDQ_MIN_I, j + ODDQ_MIN_J]))
        if len(active_coords) > 0:
            tm.set_active(active_coords)
        return tm


def play_life_hex(tmap, rule_config):
    """
    RULE_CONFIGURATION = {
        'b': (2,),  # birth
        's': (1, 2,),  # survival
    }
    """
    tocheck = set()
    for active in tmap.keys():
        tocheck.add(active)
        tocheck.update(list(map(tuple, get_ring(active, 1))))
    tocheck = list(tocheck)
    nneighbors = [len(tmap[get_ring(coord, 1)]) for coord in tocheck]
    isactive = [coord in tmap for coord in tocheck]
    newmap = TonnetzMap()
    for i in range(len(tocheck)):
        if isactive[i] and nneighbors[i] in rule_config["s"]:
            newmap.set_active(tocheck[i])
        elif not isactive[i] and nneighbors[i] in rule_config["b"]:
            newmap.set_active(tocheck[i])
    return newmap


def save_gif(frames, fpath, speed=100):
    file_name = fpath
    with open(file_name, "wb") as file:
        frames[0].save(
            file,
            append_images=frames[1:],
            duration=speed,
            loop=0,
            optimize=True,
            save_all=True
        )
    print(f"Gif saved as '{file_name}'")


def maps2tonnetzgif(tmaps, fpath, speed=100, **kwargs):
    imgs = [g.draw(**kwargs) for g in tmaps]
    save_gif(imgs, fpath, speed)


def maps2chordscore(tmaps, quarterLength=1.0, bpm=120):
    allchords = []
    for i, gen in enumerate(tmaps):
        harmony_notes = []
        for tile in gen.values():
            hnote = m21.note.Note(tile.midi_note)
            # each generation last for a whole note
            hnote.quarterLength = quarterLength
            harmony_notes.append(hnote)
        if len(harmony_notes) > 0:
            harmony_chord = m21.chord.Chord(harmony_notes)
            allchords.append(harmony_chord)
        else:
            rest = m21.note.Rest()
            rest.quarterLength = quarterLength
            allchords.append(rest)
    harmony_stream = m21.stream.Stream()
    harmony_stream.append(m21.tempo.MetronomeMark(number=bpm))
    harmony_stream.append(allchords)
    s = m21.stream.Score(id="Hex generated notes")
    p2 = m21.stream.Part(id="Chords")
    p2.append(harmony_stream)
    p2.insert(m21.instrument.Piano())
    s.insert(0, p2)
    return s


def midi_to_tonnetzmaps(midi_path, interval="quarter", midioffset=0) -> List[TonnetzMap]:
    interval2offset = {
        "eighth": 0.5,
        "quarter": 1.0,
        "half": 2.0,
        "whole": 4.0
    }
    interval = interval2offset[interval.lower()]
    score = m21.converter.parse(midi_path)
    wholesongmaps = []
    for m in score.parts[0]:
        midivals = set()
        curroffset = 0
        for mobj in m.flatten():
            if mobj.offset >= curroffset + interval:
                measuremap = TonnetzMap()
                if len(midivals) > 0:
                    measuremap.set_active_midi(list(midivals))
                    wholesongmaps.append(measuremap)
                midivals = set()
                curroffset += interval
            if isinstance(mobj, m21.chord.Chord):
                midivals.update([chnote.pitch.midi + midioffset for chnote in mobj.notes])
            elif isinstance(mobj, m21.note.Note):
                midivals.add(mobj.pitch.midi + midioffset)
        # notes = list(m.flatten().notes)
        # midivals = set()
        # for n in notes:
        #     if isinstance(n, m21.chord.Chord):
        #         midivals.update([chnote.pitch.midi for chnote in n.notes])
        #     else:
        #         midivals.add(n.pitch.midi)
        # midivals = list(midivals)
        # measuremap = TonnetzMap()
        # measuremap.set_active_midi(midivals)
        # wholesongmaps.append(measuremap)
    return wholesongmaps

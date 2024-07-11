import re

import numpy as np

__all__ = ["load"]


def load(filename):
    with open(filename, "r") as f:
        i = 0
        active = -1
        end_site = False
        names = []
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)
        for line in f:
            if "HIERARCHY" in line:
                continue
            if "MOTION" in line:
                continue
            rmatch = re.match(r"ROOT (\w+:?\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = len(parents) - 1
                continue
            if "{" in line:
                continue
            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue
            offmatch = re.match(
                r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
            )
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))])
                continue
            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                continue
            jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)  # mixamo mod
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = len(parents) - 1
                continue
            if "End Site" in line:
                end_site = True
                continue
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                positions = np.zeros((fnum, 3))
                rotations = np.zeros((fnum, len(names), 3))
                continue
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue
            dmatch = line.strip().split()
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                if channels == 3:
                    positions[i] = data_block[0:3]
                    rotations[i, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[i] = data_block[:, 0:3]
                    rotations[i, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[i] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[i, 1:] = data_block[:, 3:6]
                else:
                    raise Exception("Too many channels! %i" % channels)
                i += 1
    return positions, rotations, offsets, parents, names, frametime

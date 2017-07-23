#!/share/apps/python/2.7.10/bin/python
"""
This code takes the eigenvectors of the dynamical matrix, and calculates the
displacements of all the atoms in the supercell.

Gets the q-vectors from the file harmonics (the output of the phon code with
IPRINT=3) and the eigenvectors themselves to calculate:

u_l,s,alpha ~ epsilon_alpha,s(q-vector)exp(iq-vector.(R_l + tau_s) + c.c

R_l + tau_s is the equlibrium position of atom s in primitive cell l, tau_s is
the basis vector.

You need the ph.dyn file, and the scf.out file.

You can supply an overall phase to add in radian, or just use a random one,
(default).

Notes about the phase for 3x3 cell:
central atom strictly zero, basically equivalent to phase = 0, 2pi/3:
phase = 4.0 * np.pi / 3.0
all moving directly towards atom 1 (central):
phase = np.pi / 2.0
all moving directly away from atom 1 (central):
phase = - np.pi / 2.0

It seems to break if you rotate to the right (supplying the code an angle < 0).

"""
import sys
import getopt
import os.path
import re
import subprocess
import cmath as cm
import math as m
import numpy as np
import random as rand

# global variables
RADIAN_120 = 2.0 * np.pi / 3.0
RADIAN_060 = 1.0 * np.pi / 3.0
BOHR_TO_ANGSTROM = 0.52917721092

class a_vector:
    def __init__(self, x, y, z):
        self.v = np.array([x, y, z])
    def __rotate__(self, radian):
        m = np.array([[np.cos(radian), -np.sin(radian), 0.0],
                      [np.sin(radian),  np.cos(radian), 0.0],
                      [           0.0,             0.0, 1.0]])
        self.v = np.dot(m, self.v)
    def __print__(self, n, celldm):
        vec = np.array([self.v[0], self.v[1], self.v[2]]) * n * celldm[0]
        print "%13.9f %13.9f %13.9f" %(vec[0], vec[1], vec[2])

class atom:
    def __init__(self, name, x, y, z):
        self.name = name
        self.r = np.array([x, y, z])
    def __rotate__(self, radian):
        m = np.array([[np.cos(radian), -np.sin(radian), 0.0],
                      [np.sin(radian),  np.cos(radian), 0.0],
                      [           0.0,             0.0, 1.0]])
        self.r = np.dot(m, self.r)
    def __printXSF__(self, disp, celldm, opt_list):
        if opt_list.calc is 'xsf':
            pos = self.r * celldm[0]
            shift = disp.real
            shift_size = np.linalg.norm(shift) * celldm[0]
            shift = disp.real / opt_list.n[0]
            print ("%3s %13.9f %13.9f %13.9f %13.9f %13.9f %13.9f "
                   "# %4.3f Angstrom"
                   %(self.name, pos[0], pos[1], pos[2],
                                shift[0], shift[1], shift[2], shift_size))
        else:
            pos = self.r / opt_list.n[0]
            shift = np.array([disp[0].real, disp[1].real, disp[2].real])
            shift_size = np.linalg.norm(shift) * celldm[0]
            shift /= opt_list.n[0]
            shifted = pos + shift
            print ("%3s %13.9f %13.9f %13.9f %13.9f %13.9f %13.9f "
                   "%13.9f %13.9f %13.9f # %4.3f"
                  %(self.name, pos[0], pos[1], pos[2],
                               shift[0], shift[1], shift[2],
                               shifted[0], shifted[1], shifted[2], shift_size))
    def __printCRYSTL__(self, disp, celldm, inverted_a, opt_list):
        shift = np.array([disp[0].real, disp[1].real, disp[2].real])
        shift_size = np.linalg.norm(shift) * celldm[0]
        shifted = self.r / opt_list.n
        shift /= opt_list.n
        if opt_list.noshift is False:
            shifted += shift
        shifted = np.dot(inverted_a, shifted)
        print ("%3s %13.9f %13.9f %13.9f"
                %(self.name, shifted[0], shifted[1], shifted[2]))

class eigen:
    def __init__(self, q1, q2, q3, mode, freq, er1, ei1, er2, ei2, er3, ei3):
        self.q = np.array([q1, q2, q3])
        self.mode = mode
        self.freq = freq
        self.eigenvector = np.array([complex(er1, ei1), complex(er2, ei2),
                                     complex(er3, ei3)])
        self.phase = cm.phase(self.eigenvector[0])
    def __rotate__(self, radian):
        m = np.array([[np.cos(radian), -np.sin(radian), 0.0],
                      [np.sin(radian),  np.cos(radian), 0.0],
                      [           0.0,             0.0, 1.0]])
        self.q = np.dot(m, self.q)
        self.eigenvector = np.dot(m, self.eigenvector)

def __calculation__(eigenvector, atom, celldm, opt_list, inverted_a):
    displacements = [complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0)]
    # factor = celldm[0] * opt_list.factor
    eigenvector.__rotate__(RADIAN_060)
    for i in range(0, 3):
        eigenvector.__rotate__(RADIAN_120)
        argument = 2.0 * np.pi * eigenvector.q.dot(atom.r) \
                       - eigenvector.phase + opt_list.phase[i]
        exponential = complex(np.cos(argument), np.sin(argument))
        displacements += eigenvector.eigenvector * exponential * opt_list.factor
    eigenvector.__rotate__(-RADIAN_060)
    if opt_list.calc is 'crys':
        atom.__printCRYSTL__(displacements, celldm, inverted_a, opt_list)
    else:
        atom.__printXSF__(displacements, celldm, opt_list)

def __invert__(a):
    m = np.array([[a[0].v[0], a[1].v[0], a[2].v[0]],
                  [a[0].v[1], a[1].v[1], a[2].v[1]],
                  [a[0].v[2], a[1].v[2], a[2].v[2]]])
    return np.linalg.inv(m)

class options:
    def __init__(self):
        self.n_x = self.n_y = self.n_z = 1
        self.n = [self.n_x, self.n_y, self.n_z]
        self.scf_out = 'scf.out'
        self.dyn_file = 'ph.dyn'
        self.rotation_angle = 0.0
        self.calc = 'xsf'
        rand.seed()
        self.r_phase = rand.random() * 2.0 * np.pi
        self.phase = [0.0, 0.0, 0.0]
        self.noshift = False
        self.rotate = False
        self.user_defined_shift = [0.0, 0.0, 0.0]
        self.factor = 0.005
        self.animsteps = 1

def __usage__(return_val):
    message = ["USAGE:\n",
            "\tespresso-harmonic.py -x n_x -y n_y -z n_z -s scf.out -d "
            "modes.dyn [-p <phase>] "
            "[-L <G1_phase>] [-M <G2_phase>] [-N <G3_phase>] "
            "[-i shift_x -j shift_y -k shift_z] "
            "[-c] [-X] [-C] [-n] [-r <angle>] [-f factor] [-a animsteps]\n",
            "n_x n_y n_z:\n",
            "\tdimensions of supercell\n",
            "scf.out:\n",
            "\tespresso output with unit vectors\n",
            "modes.dyn:\n",
            "\tdynmat output with eigenvectors\n",
            "phases (by default phase is zero, \"random\" gives a "
            "random phase):\n",
            "\tfloating point number with desired phase in "
            "radian (or \"random\")\n",
            "shift_x shift_y shift_z:\n",
            "\tamount to shift atoms from center "
            "(in Cartesian, units of alat)\n",
            "-c, --crystal:\n",
            "\tgives displaced crystal coordinates suitable for scf.in\n",
            "-X, --xsf:\n",
            "\tGives an XCrysDen xsf file (this is default)\n",
            "-C, --cart:\n",
            "\tGives just atomic positions in alat (Cartesian) units.\n",
            "-n, --no_shift:\n",
            "\tGives an undistorted structure\n",
            "angle:\n",
            "\tThe angle by which to rotate the cell (in radian)\n",
            "\tIf defined, n_x, n_y, & n_z take on a slightly different ",
            "meaning\n\tThey refer to the dimensions of the supercell ",
            "in crystal coordinates\n\tAngle must be < 0\n",
            "factor:\n\tOverall amplitude factor\n",
            "animsteps:\n\tNumber of modes to include. Useful for when",
            " the mode of interest is not the lowest."]
    print ''.join(message)
    sys.exit(return_val)

def main(argv):
    """
        main function
    """
    # rand.seed()
    # phase = rand.random() * 2.0 * np.pi
    # print "# %13.9f" %(phase)
    try:
        opts, args = getopt.getopt(argv, "hcXnCx:y:z:s:d:p:i:j:k:r:f:a:L:M:N:",
                                        ["help", "crystal", "xcrysden", "cart",
                                         "no_shift", "n_x=", "n_y", "n_z",
                                         "scf_out=", "dyn_file=", "phase=",
                                         "shift_x=", "shift_y=", "shift_z=",
                                         "rotation_angle=", "factor=",
                                         "animsteps=", "G1_phase=",
                                         "G2_phase", "G3_phase"])
    except getopt.GetoptError:
        __usage__(2)
    if opts == []:
        __usage__(1)
    opt_list = options()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            __usage__(0)
        elif opt in ("-c", "--crystal"):
            opt_list.calc = 'crys'
        elif opt in ("-C", "--cart"):
            opt_list.calc = 'cart'
        elif opt in ("-X", "--xcrysden"):
            opt_list.calc = 'xsf'
        elif opt in ("-x", "--n_x"):
            opt_list.n_x = int(arg)
        elif opt in ("-y", "--n_y"):
            opt_list.n_y = int(arg)
        elif opt in ("-z", "--n_z"):
            opt_list.n_z = int(arg)
        elif opt in ("-s", "--scf_out"):
            opt_list.scf_out = arg
        elif opt in ("-d", "--dyn_file"):
            opt_list.dyn_file = arg
        elif opt in ("-p", "--phase"):
            if arg == "random":
                phase = opt_list.r_phase
                opt_list.phase = [phase, phase, phase]
            else:
                opt_list.phase = [float(arg), float(arg), float(arg)]
        elif opt in ("-L", "--G1_phase"):
            if arg == "random":
                opt_list.phase[0] = opt_list.r_phase
            else:
                opt_list.phase[0] = float(arg)
        elif opt in ("-M", "--G2_phase"):
            if arg == "random":
                opt_list.phase[1] = opt_list.r_phase
            else:
                opt_list.phase[1] = float(arg)
        elif opt in ("-N", "--G3_phase"):
            if arg == "random":
                opt_list.phase[2] = opt_list.r_phase
            else:
                opt_list.phase[2] = float(arg)
        elif opt in ("-n", "--no_shift"):
            opt_list.noshift = True
        elif opt in ("-f", "--factor"):
            opt_list.factor *= float(arg)
        elif opt in ("-i", "--shift_x"):
            opt_list.user_defined_shift[0] = float(arg)
        elif opt in ("-j", "--shift_y"):
            opt_list.user_defined_shift[1] = float(arg)
        elif opt in ("-k", "--shift_z"):
            opt_list.user_defined_shift[2] = float(arg)
        elif opt in ("-r", "--rotation_angle"):
            opt_list.rotation_angle = float(arg)
            if rotation_angle < 0.0:
                print "Please use an angle > 0."
                __usage__(5)
            opt_list.rotate = True
        elif opt in ("-a", "--animsteps"):
            opt_list.animsteps = int(arg)
    if args != []:
        print 'these are superfluous args: ', args
        __usage__(2)
    collection = [opt_list.scf_out, opt_list.dyn_file]
    for i in collection:
        input = i
        if os.path.isfile(input) == False:
            print input, " does not seem to exist."
            __usage__(4)

    num_cells = opt_list.n_x * opt_list.n_y * opt_list.n_z
    opt_list.n = [opt_list.n_x, opt_list.n_y, opt_list.n_z]
    if opt_list.rotate is True:
        num_cells += 1
        opt_list.n[0] = opt_list.n[1] = m.sqrt(num_cells)

    a = []
    atoms = []
    with open(opt_list.scf_out, 'r') as f:
        started = False
        lines = [l.strip() for l in f]
        for line in lines:
            if 'celldm(1)' in line:
                fields = line.split()
                celldm = [float(fields[1]), float(fields[5])]
                celldm = [BOHR_TO_ANGSTROM * x for x in celldm]
            if re.search('a\([1-3]\)', line):
                fields = line.split()
                a.append(a_vector(float(fields[3]),
                                  float(fields[4]),
                                  float(fields[5])))
            if re.search('positions \(alat units\)', line):
                started = True
            if re.search('^$', line) and started is True:
                started = False
            if started is True and re.search('tau\(', line):
                fields = line.split()
                for i in range(0, opt_list.n_x):
                    for j in range(0, opt_list.n_y):
                        for k in range(0, opt_list.n_z):
                            r_shift = (i * a[0].v +
                                       j * a[1].v +
                                       k * a[2].v +
                                       opt_list.user_defined_shift)
                            atoms.append(atom(fields[1],
                                         r_shift[0] + float(fields[6]),
                                         r_shift[1] + float(fields[7]),
                                         r_shift[2] + float(fields[8])))
                            if (opt_list.calc is 'crys' 
                                and opt_list.rotate is True):
                                current = len(atoms) - 1
                                atoms[current].__rotate__(-rotation_angle)
                if opt_list.rotate is True:
                    if opt_list.n_x > opt_list.n_y:
                        r_shift = (opt_list.n_x * a[0].v +
                                   0 * a[1].v +
                                   0 * a[2].v +
                                   opt_list.user_defined_shift)
                    else:
                        r_shift = (0 * a[0].v +
                                   opt_list.n_y * a[1].v +
                                   0 * a[2].v +
                                   opt_list.user_defined_shift)
                    atoms.append(atom(fields[1],
                                 r_shift[0] + float(fields[6]),
                                 r_shift[1] + float(fields[7]),
                                 r_shift[2] + float(fields[8])))
                    if opt_list.calc is 'crys':
                        current = len(atoms) - 1
                        atoms[current].__rotate__(-opt_list.rotation_angle)

    num_atoms = len(atoms) / num_cells
    count = 0
    maxCount = opt_list.animsteps * num_atoms
    inverted_a = []
    if opt_list.calc is 'crys':
        inverted_a = __invert__(a)
    elif opt_list.rotate is True:
        for i in range(0, len(a)):
            a[i].__rotate__(opt_list.rotation_angle)

    with open(opt_list.dyn_file, 'r') as f:
        started = False
        mode = 'tmp'
        freq = 'tmp'
        lines = [l.strip() for l in f]
        for line in lines:
            if re.search('Diagonalizing', line):
                started = True
            if started is True:
                if re.search('q = \(', line):
                    fields = line.split()
                    q = [float(fields[3]), float(fields[4]), float(fields[5])]
                    if opt_list.calc is 'xsf':
                        # print "# q-vector:", q[0], q[1], q[2]
                        if opt_list.animsteps > 1:
                            print "ANIMSTEPS ", maxCount / num_atoms
                        print "CRYSTAL"
                        # print "SLAB"
                        print "PRIMVEC"
                        for i in range(0, 3):
                            a[i].__print__(opt_list.n[i], celldm)
                        # print "CONVVEC"
                        # for i in range(0, 3):
                            # a[i].__print__(opt_list.n[i], celldm)
                elif re.search('freq \(', line):
                    fields = line.split()
                    mode = fields[2].strip(')')
                    freq = fields[7]
                    if opt_list.calc is 'xsf':
                        # print "# mode ", mode, "# eigenvalue ", freq
                        print "PRIMCOORD ", mode
                        print num_atoms * num_cells, " 1"
                    count = 0
                elif re.search('[0-9]', line):
                    fields = line.split()
                    evec = eigen(q[0], q[1], q[2], mode, freq,
                                 float(fields[1]), float(fields[2]),
                                 float(fields[3]), float(fields[4]),
                                 float(fields[5]), float(fields[6]))
                    if opt_list.calc is 'crys' and opt_list.rotate is True:
                            evec.__rotate__(-opt_list.rotation_angle)
                    for i in range(0, num_cells):
                        __calculation__(evec,
                                        atoms[i + count * num_cells],
                                        celldm, opt_list, inverted_a)
                    count += 1
                if count == maxCount:
                    break
    sys.stdout.close()

if __name__ == "__main__":
    main(sys.argv[1:])

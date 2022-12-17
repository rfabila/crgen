import random
import math


import PyDCG

turn = PyDCG.geometricbasics.turn
LEFT = PyDCG.geometricbasics.LEFT
RIGHT = PyDCG.geometricbasics.RIGHT


def segsIntersect(s1, s2):
    if s1[0] in s2 or s1[1] in s2:
        return False
    return turn(s1[0], s1[1], s2[0]) != turn(s1[0], s1[1], s2[1]) and turn(s2[0], s2[1], s1[0]) != turn(s2[0], s2[1], s1[1])


def randomGeometricGraph(n, p=1/3):
    pts = [PyDCG.datastructures.randPoint() for i in range(n)]
    G = {x: [] for x in range(n)}
    segs = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() <= p:
                segs.append([pts[i], pts[j]])
                G[i].append(j)
                G[j].append(i)
    return G, pts, segs


def naiveCr(G, pts):
    cr = 0
    edges = set()
    for u in G:
        for v in G[u]:
            edges.add((u, v) if u <= v else (v, u))
    edges = list(edges)
    for i in range(len(edges)):
        e = edges[i]
        for j in range(i+1, len(edges)):
            f = edges[j]
            p1, p2 = map(lambda x: pts[x], e)
            q1, q2 = map(lambda x: pts[x], f)
            if segsIntersect((p1, p2), (q1, q2)):
                cr += 1
    return cr


class node:
    def __init__(self, data, isRoot=False, isLeft=False, isRight=False, parent=None, rson=None, lson=None, isLeaf=False):
        self.data = data
        self.isRoot = isRoot
        self.isLeft = isLeft
        self.isRight = isRight
        self.parent = parent
        self.rson = rson
        self.lson = lson
        self.isLeaf = isLeaf

    # def __eq__(self, other):
    #     return self.data == other.data and self.isRoot == other.isRoot and self.rson == other.rson and self.lson == other.lson

    def find(self, data):
        # print("searching...", self.data)
        if self.data == data:
            return self

        res = None

        if self.lson is not None:
            res = self.lson.find(data)
            # print("lson", res)
            if res is not None:
                # print("found!")
                return res

        if self.rson is not None:
            res = self.rson.find(data)
            # print("rson", res)

        # print("ret", res)
        return res

    def print(self):
        if self.isLeaf:
            print(self.data)
        else:
            self.lson.print()
            self.rson.print()
            print(self.data)


def quadcr(oG, opts):
    # pts = sorted(opts, key=lambda p: p[1], reverse=True)
    pts = [tuple(p[:]) for p in opts]
    pts.sort(key=lambda p: p[1], reverse=True)
    # print(pts)
    n = len(pts)

    G = {p: [] for p in pts}

    for p in pts:
        oidx = opts.index(list(p))
        oneighbors = oG[oidx]
        for idx in oneighbors:
            q = opts[idx]
            G[p].append(tuple(q))

    def getAnti(p):
        # anti = set()
        anti = []
        for q in G[p]:
            if p == q:
                continue
            anti.append((2*p[0]-q[0], 2*p[1]-q[1]))
        return anti

    def getAntiFull(p):
        anti = set()
        for q in pts:
            if p == q:
                continue
            anti.add((2*p[0]-q[0], 2*p[1]-q[1]))
        return anti

    antipodals = {tuple(p): getAnti(p) for p in pts}

    #                                              Step 1
    sortedPointsAnti = [PyDCG.geometricbasics.sort_around_point(pts[i], tuple(pts[:i] + pts[i+1:] + antipodals[pts[i]])) for i in range(len(pts))]
    sortedPoints = [list(filter(lambda p: p in pts, spts)) for spts in sortedPointsAnti]

    #                                              Step 2
    # print(oSortedPoints, sortedPoints)
    vwPlus = {}
    vwMinus = {}

    for i in range(n):
        w = pts[i]
        Nw = len(G[w])
        # print("w", w)
        # print("neighbors", G[w])
        # print("anti", antipodals[w])
        ri = 0
        # Calculate r0
        v0 = sortedPointsAnti[i][0]
        for u in sortedPointsAnti[i][1:]:
            if u in antipodals[w]:
                continue
            if u in G[w] and turn(w, v0, u) == RIGHT:
                ri += 1
        # print("r0", ri)
        # print("v0", v0)
        # assert ri >= 0
        # assert ri <= Nw

        if v0 not in antipodals[w]:
            vwPlus[(v0, w)] = ri
            vwMinus[(v0, w)] = Nw - ri - (1 if v0 in G[w] else 0)
        else:
            ri += 1

        vprev = v0
        for vi in sortedPointsAnti[i][1:]:
            # print("checking", vi)
            if vprev in antipodals[w]:  # v_{i-1} is red
                # print("anti")
                ri = max(0, ri-1)
            else:
                if vprev in G[w]:  # v_{i-1} is blue
                    # print("neighbor")
                    ri += 1
                # else:
                    # print("gray")
            if vi not in antipodals[w]:
                vwPlus[(vi, w)] = ri
                vwMinus[(vi, w)] = Nw - ri - (1 if vi in G[w] else 0)
                # print("ri", ri)
            vprev = vi

    # for k, v in vwPlus.items():
    #     print(k, v)

    # Sanity check, remove
    # for p in pts:
    #     for q in pts:
    #         if p == q:
    #             continue
    #         left = 0
    #         for r in pts:
    #             if r == p or r == q:
    #                 continue
    #             if r in G[q] and turn(p, q, r) == LEFT:
    #                 left += 1

    #         if vwPlus[(p, q)] != left:
    #             print("Wrong value vw plus!")
    #             print((p, q), vwPlus[(p, q)], left)
    #             # raise Exception
    # for p in pts:
    #     for q in pts:
    #         if p == q:
    #             continue
    #         left = 0
    #         for r in pts:
    #             if r == p or r == q:
    #                 continue
    #             if r in G[q] and turn(p, q, r) == RIGHT:
    #                 left += 1

    #         if vwMinus[(p, q)] != left:
    #             print("Wrong value vw minus")
    #             print((p, q), vwMinus[(p, q)], left)
                # raise Exception
    # End sanity check

    #                                           Step 3

    lv = [0 for i in range(n)]
    lvb = [0 for i in range(n)]
    trees = []

    def buildTree(p, down):
        # print("building", p, down)
        # print("N", G[p])
        height = math.ceil(math.log(len(down), 2))
        leaves = 2**height
        processingLeaves = True
        currentNodes = [node(data=None, isLeaf=True) for i in range(leaves)]
        # tree = node(data=[0,0], isRoot = True)
        for i in range(leaves):
            if i < len(down):
                currentNodes[i].data = down[i]
            else:
                currentNodes[i].data = None

            if i % 2 == 0:
                currentNodes[i].isLeft = True
            else:
                currentNodes[i].isRight = True

        while len(currentNodes) > 1:
            upperLevel = []
            for i in range(0, len(currentNodes), 2):
                data = [0, 0]
                if processingLeaves:
                    data[0] = 1 if (currentNodes[i].data is not None and currentNodes[i].data in G[p]) else 0
                    data[1] = 1 if (currentNodes[i+1].data is not None and currentNodes[i+1].data in G[p]) else 0
                else:
                    data[0] = sum(currentNodes[i].data)
                    data[1] = sum(currentNodes[i+1].data)
                parent = node(data, lson=currentNodes[i], rson=currentNodes[i+1])
                if len(upperLevel) % 2 == 0:
                    parent.isLeft = True
                else:
                    parent.isRight = True
                currentNodes[i].parent = parent
                currentNodes[i+1].parent = parent
                upperLevel.append(parent)
            if processingLeaves:
                processingLeaves = False
            assert len(upperLevel) == len(currentNodes)/2
            currentNodes = upperLevel
        currentNodes[0].isRoot = True
        return currentNodes[0]

    for i in range(n-1):
        down = [p for p in sortedPoints[i] if p[1] < pts[i][1]]
        # print("srted for", pts[i])
        # print(down)
        for q in down:          # Do this right
            good = True
            for r in down:
                if q == r:
                    continue
                if turn(pts[i], q, r) == RIGHT:
                    # print(q, "is bad")
                    good = False
                    break
            if good:
                # print(q, "is good")
                break
        k = down.index(q)
        aux = [None for i in range(len(down))]
        for j in range(len(aux)):
            aux[j] = down[k % len(down)]
            k += 1
        down = aux
        # print("for", pts[i], "leftmost is", q)
        # print("p, down", pts[i], down)
        trees.append(buildTree(pts[i], down))

    def ptsToTheRight(leaf):
        res = 0
        aux = leaf
        while not aux.parent.isRoot:
            # print("ptsRight", aux.data)
            if aux.isLeft:
                # print(aux.data, "isLeft, adding", aux.parent.data[1])
                res += aux.parent.data[1]
            aux = aux.parent
        if aux.isLeft:
            # print(aux.data, "isLeft, adding", aux.parent.data[1])
            res += aux.parent.data[1]
        return res

    def ptsToTheLeft(leaf):
        res = 0
        aux = leaf
        while not aux.parent.isRoot:
            # print("ptsRight", aux.data)
            if aux.isRight:
                # print(aux.data, "isLeft, adding", aux.parent.data[1])
                res += aux.parent.data[0]
            aux = aux.parent
        if aux.isRight:
            # print(aux.data, "isLeft, adding", aux.parent.data[1])
            res += aux.parent.data[0]
        return res

    def updateTree(leaf):
        aux = leaf
        while not aux.parent.isRoot:
            # print("updating", aux.data)
            if aux.isLeft:
                if aux.parent.data[0] == 0:
                    return
                else:
                    aux.parent.data[0] -= 1
            if aux.isRight:
                if aux.parent.data[1] == 0:
                    return
                else:
                    aux.parent.data[1] -= 1
            aux = aux.parent
        if aux.isLeft:
            if aux.parent.data[0] == 0:
                return
            else:
                aux.parent.data[0] -= 1
        if aux.isRight:
            if aux.parent.data[1] == 0:
                return
            else:
                aux.parent.data[1] -= 1

    for i in range(1, n-1):
        for j in range(i):
            leaf = trees[j].find(pts[i])
            assert leaf is not None
            # print(i, j)
            # print("to the right", ptsToTheRight(leaf))
            lv[i] += ptsToTheRight(leaf)
            lvb[i] += ptsToTheLeft(leaf)
            updateTree(leaf)

    # print(lv)
    # print(lvb)

    #                                           Step 4

    alphavw = {}
    betavw = {}

    antipodals = {tuple(p): getAntiFull(p) for p in pts}
    sortedPointsAnti = [PyDCG.geometricbasics.sort_around_point(pts[i], tuple(pts[:i] + pts[i+1:] + list(antipodals[pts[i]]))) for i in range(len(pts))]

    for i in range(n):
        v = pts[i]
        aux = [v[0]+1, v[1]]
        # For alpha
        pointsAbove = False
        for q in sortedPoints[i]:
            if q[1] > aux[1]:
                pointsAbove = True
        if pointsAbove:
            for j in range(len(sortedPoints[i])):
                q = sortedPoints[i][j]
                if turn(v, aux, q) == LEFT:
                    break
        else:
            for q in sortedPoints[i]:          # Do this right
                good = True
                for r in sortedPoints[i]:
                    if q == r:
                        continue
                    if turn(pts[i], q, r) == RIGHT:
                        # print(q, "is bad")
                        good = False
                        break
                if good:
                    # print(q, "is good")
                    break
            j = sortedPoints[i].index(q)
        widx = j
        lastAlpha = lv[i]
        lastvwPlus = 0
        m = len(sortedPoints[i])
        for j in range(m):
            wi = sortedPoints[i][widx % m]
            alphavw[(v, wi)] = lastAlpha + lastvwPlus - vwMinus[(v, wi)]
            lastAlpha = alphavw[(v, wi)]
            lastvwPlus = vwPlus[(v, wi)]
            widx += 1

        # For beta
        # print("v is", v)
        down = [p for p in sortedPointsAnti[i] if p[1] < aux[1]]

        for q in down:          # Do this right
            good = True
            for r in down:
                if q == r:
                    continue
                if turn(pts[i], q, r) == RIGHT:
                    # print(q, "is bad")
                    good = False
                    break
            if good:
                # print(q, "is good")
                break
        j = sortedPointsAnti[i].index(q)
        # print("down", down)
        # print("j for", v, j)

        widx = j
        lastBeta = lvb[i]
        lastvwPlus = 0
        m = len(sortedPointsAnti[i])
        for j in range(m):
            wi = sortedPointsAnti[i][widx % m]
            if wi in antipodals[v]:
                # print("wi is antipodal", wi)
                w = (2*v[0]-wi[0], 2*v[1]-wi[1])
                lastBeta = lastBeta + lastvwPlus  # - vwMinus[(v, wi)]
                lastvwPlus = 0
                betavw[(v, w)] = lastBeta
                # print("beta", pts.index(v), pts.index(w), "is", betavw[(v, w)])
            else:
                # print("wi is a real point", wi)
                # print("lastbeta from", lastBeta)
                lastBeta = lastBeta + lastvwPlus - vwMinus[(v, wi)]
                # lastBeta = betavw[(v, wi)]
                lastvwPlus = vwPlus[(v, wi)]
                # print("to", lastBeta)
            widx += 1

    # Check alphas, remove
    # for i in range(n):
    #     v = pts[i]
    #     for j in range(n):
    #         if i == j:
    #             continue
    #         w = pts[j]
    #         alpha = 0
    #         for u in G:
    #             if u in (v, w):
    #                 continue
    #             for x in G[u]:
    #                 if x in (v, w):
    #                     continue
    #                 if turn(v, w, u) != turn(v, w, x):
    #                     l1 = PyDCG.line.Line(v, w)
    #                     l2 = PyDCG.line.Line(u, x)
    #                     intersection = tuple(l1.intersection(l2))
    #                     if (v < w and v < intersection) or (v > w and v > intersection):
    #                         alpha += 1
    #         if alpha % 2 == 1 or alpha//2 != alphavw[(v, w)]:
    #             print("Different values", v, w)
    #             print(alphavw[(v, w)], "vs", alpha/2)
            # else:
            #     print("Same!", alpha/2, alphavw[(v,w)])
    # End checkalphas

    # Check betas, remove
    # for i in range(n):
    #     v = pts[i]
    #     for j in range(n):
    #         if i == j:
    #             continue
    #         w = pts[j]
    #         beta = 0
    #         for u in G:
    #             if u in (v, w):
    #                 continue
    #             for x in G[u]:
    #                 if x in (v, w):
    #                     continue
    #                 if turn(v, w, u) != turn(v, w, x):
    #                     l1 = PyDCG.line.Line(v, w)
    #                     l2 = PyDCG.line.Line(u, x)
    #                     intersection = tuple(l1.intersection(l2))
    #                     if (v < w and intersection < v) or (v > w and intersection > v):
    #                         beta += 1
    #         if beta % 2 == 1 or beta//2 != betavw[(v, w)]:
    #             print("Different values", v, w)
    #             print(betavw[(v, w)], "vs", beta/2)
            # else:
            #     print("Same!", alpha/2, alphavw[(v,w)])
    # End checkalphas

    alpha = sum(alphavw[(v, w)] for v in pts for w in G[v])
    beta = sum(betavw[(v, w)] for v in pts for w in G[v])

    return (alpha-beta)//4


def buildG(pts, segs):
    G = {i: [] for i in range(len(pts))}
    for e in segs:
        u, v = map(pts.index, e)
        G[u].append(v)
        G[v].append(u)
    return G

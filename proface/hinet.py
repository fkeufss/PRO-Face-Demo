# from model import *
from torch import nn
from invblock import INV_block, INV_block_affine


class Hinet(nn.Module):

    def __init__(self):
        super(Hinet, self).__init__()

        # self.inv1 = INV_block()
        # self.inv2 = INV_block()
        # self.inv3 = INV_block()
        # self.inv4 = INV_block()
        # self.inv5 = INV_block()
        # self.inv6 = INV_block()
        # self.inv7 = INV_block()
        # self.inv8 = INV_block()

        self.inv1 = INV_block_affine()
        self.inv2 = INV_block_affine()
        self.inv3 = INV_block_affine()
        self.inv4 = INV_block_affine()
        self.inv5 = INV_block_affine()
        self.inv6 = INV_block_affine()
        # self.inv7 = INV_block_affine()
        # self.inv8 = INV_block_affine()

        # self.inv9 = INV_block()
        # self.inv10 = INV_block()
        # self.inv11 = INV_block()
        # self.inv12 = INV_block()
        # self.inv13 = INV_block()
        # self.inv14 = INV_block()
        # self.inv15 = INV_block()
        # self.inv16 = INV_block()

    def forward(self, x, password, rev=False):
        if not rev:
            out = self.inv1(x, password)
            out = self.inv2(out, password)
            out = self.inv3(out, password)
            out = self.inv4(out, password)
            out = self.inv5(out, password)
            out = self.inv6(out, password)
            # out = self.inv7(out, password)
            # out = self.inv8(out, password)

            # out = self.inv9(out)
            # out = self.inv10(out)
            # out = self.inv11(out)
            # out = self.inv12(out)
            # out = self.inv13(out)
            # out = self.inv14(out)
            # out = self.inv15(out)
            # out = self.inv16(out)
        else:
            # out = self.inv16(x, rev=True)
            # out = self.inv15(out, rev=True)
            # out = self.inv14(out, rev=True)
            # out = self.inv13(out, rev=True)
            # out = self.inv12(out, rev=True)
            # out = self.inv11(out, rev=True)
            # out = self.inv10(out, rev=True)
            # out = self.inv9(out, rev=True)

            # out = self.inv8(x, password, rev=True)
            # out = self.inv7(out, password, rev=True)
            out = self.inv6(x, password, rev=True)
            out = self.inv5(out, password, rev=True)
            out = self.inv4(out, password, rev=True)
            out = self.inv3(out, password, rev=True)
            out = self.inv2(out, password, rev=True)
            out = self.inv1(out, password, rev=True)

        return out



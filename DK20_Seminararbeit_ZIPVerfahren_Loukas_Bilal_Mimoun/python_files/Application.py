"""
    Lempel-Ziv Prinzip wurde von uns:
        Bilal Boui
        Loukas Melissopoulos
        Mimoun El Masiani
    programiert.

    Lempel-Ziv Welch wurde von http://rosettacode.org/wiki/LZW_compression entnommen.
    Dies unterliegt der GNU-Lizenz für freie Dokumentation (GNU Free Documentation License)

    In diesem Program muss gewählt werden, welches der beiden Kompressionsalgorithmen benutzt werden soll.
"""

import Lempel_Ziv
import Lempel_ZWe

from tkinter import Tk, Frame, Button


class Main(Tk):

    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.parent = parent

        # WINDOW SETTINGS
        self.title('Lempel - Ziv Algorithms')
        self.resizable(0, 0)

        self.universe = Frame(self)
        self.universe.pack()

        self.frame_btn = Frame(self.universe)
        self.frame_btn.grid(row=0, column=0)

        btn__lz = Button(self.frame_btn, width=20, height=5, text='Lempel-Ziv (principle)', relief='groove',
                         command=self.lz)
        btn_lzw = Button(self.frame_btn, width=20, height=5, text='Lempel-Ziv-Welch', relief='groove',
                         command=self.lzw)
        btn__lz.grid(row=0, column=0), btn_lzw.grid(row=0, column=1)

    def lz(self):
        self.update()
        Tk.destroy(self)

        lz = Lempel_Ziv.Main(None)
        lz.eval('tk::PlaceWindow . center')
        lz.mainloop()

    def lzw(self):
        self.update()
        Tk.destroy(self)

        lzw = Lempel_ZWe.Main(None)
        lzw.eval('tk::PlaceWindow . center')
        lzw.mainloop()


if __name__ == "__main__":
    root = Main(None)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

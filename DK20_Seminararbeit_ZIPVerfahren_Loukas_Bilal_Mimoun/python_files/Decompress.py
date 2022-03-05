"""
    This is a standalone Decompressor, that takes a compressed message as input, generated from the
    Lempel-Ziv algorithm.
"""

from tkinter import Tk, Frame, Button, Label, Text, Scrollbar, VERTICAL, END, INSERT


class Main(Tk):

    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.parent = parent

        # GLOBAL VARIABLES
        self.dec_codebook = []
        self.dec_msg = []

        # WINDOW SETTINGS
        self.title("DECOMPRESSOR")
        self.resizable(0, 0)

        # FRAMES
        self.universe = Frame(self)
        self.universe.pack()

        # BUTTONS
        self.btn_frame = Frame(self.universe)
        self.btn_frame.grid(row=0, column=0)

        self.btn_start = Button(self.btn_frame, width=15, text='START', relief='groove',
                                command=self.decode)
        self.btn_clear = Button(self.btn_frame, width=15, text='CLEAR', relief='groove',
                                command=self.clear)
        self.btn__exit = Button(self.btn_frame, width=15, text='EXIT', relief='groove',
                                command=self.destroy)

        self.btn_start.grid(row=0, column=0, padx=(10, 122))
        self.btn_clear.grid(row=0, column=1, padx=(100, 100))
        self.btn__exit.grid(row=0, column=2, padx=(122, 10))

        # TEXT FIELD (INPUT/OUTPUT)
        self.frame_universe = Frame(self.universe)
        self.frame_universe.grid(row=1, column=0)

        # INPUT
        self.frame__input = Frame(self.frame_universe)
        self.frame__input.grid(row=0, column=0)

        self.lbl_frame__input = Label(self.frame__input, text='INPUT', anchor='s')
        self.input___txt__fld = Text(self.frame__input, width=96, height=7)
        self.scroll_bar_input = Scrollbar(self.frame__input, command=self.input___txt__fld.yview, orient=VERTICAL)
        self.input___txt__fld['yscrollcommand'] = self.scroll_bar_input.set
        self.input___txt__fld.see(END)

        self.lbl_frame__input.grid(row=0, column=0, sticky='ew', padx=(10, 0))
        self.input___txt__fld.grid(row=1, column=0, sticky='ew', padx=(10, 0))
        self.scroll_bar_input.grid(row=1, column=1, sticky='ns', padx=(0, 10))

        # OUTPUT
        self.frame_output = Frame(self.frame_universe)
        self.frame_output.grid(row=1, column=0)

        self.lbl__frame_output = Label(self.frame_output, text='OUTPUT', anchor='s')
        self.output___txt_fld = Text(self.frame_output, width=96, height=7)
        self.scroll_bar_output = Scrollbar(self.frame_output, command=self.output___txt_fld.yview, orient=VERTICAL)
        self.output___txt_fld['yscrollcommand'] = self.scroll_bar_output.set
        self.output___txt_fld.see(END)

        self.lbl__frame_output.grid(row=0, column=0, sticky='ew', padx=(10, 0))
        self.output___txt_fld.grid(row=1, column=0, sticky='ew', padx=(10, 0))
        self.scroll_bar_output.grid(row=1, column=1, sticky='ns', padx=(0, 10))

        # GENERATED CODEBOOK
        self.frame_codebook = Frame(self.frame_universe)
        self.frame_codebook.grid(row=2, column=0, pady=(0, 20))

        self.lbl__frame_cdbook = Label(self.frame_codebook, text='GENERATED DICTIONARY', anchor='s')
        self.cdbook___txt__fld = Text(self.frame_codebook, width=96, height=7)
        self.scroll_bar_cdbook = Scrollbar(self.frame_codebook, command=self.cdbook___txt__fld.yview, orient=VERTICAL)
        self.cdbook___txt__fld['yscrollcommand'] = self.scroll_bar_cdbook.set

        self.lbl__frame_cdbook.grid(row=0, column=0, sticky='ew', padx=(10, 0))
        self.cdbook___txt__fld.grid(row=1, column=0, sticky='ew', padx=(10, 0))
        self.scroll_bar_cdbook.grid(row=1, column=1, sticky='ns', padx=(0, 10))

    def clear(self):
        self.dec_codebook = []
        self.dec_msg = []

        self.input___txt__fld.delete('1.0', END)
        self.output___txt_fld.delete('1.0', END)
        self.cdbook___txt__fld.delete('1.0', END)

    def decode(self):
        raw_input = self.input___txt__fld.get('1.0', 'end-1c')
        array_coded_msg = raw_input.replace("']", "").replace("['", "").replace("\n", "").split("', '")
        for i in range(len(array_coded_msg)):
            if len(array_coded_msg[i]) == 1:
                if not self.dec_codebook.__contains__(array_coded_msg[i]):
                    self.dec_codebook.append((i + 1, array_coded_msg[i]))
                self.dec_msg.append(array_coded_msg[i])
            if len(array_coded_msg[i]) > 1:
                temp = array_coded_msg[i][0:-1]
                for j in range(len(self.dec_codebook)):
                    if self.dec_codebook[j][0] == int(temp):
                        self.dec_msg.append(self.dec_codebook[j][1] + array_coded_msg[i][-1])
                        if not self.dec_codebook.__contains__(self.dec_msg[-1]):
                            self.dec_codebook.append((i + 1, self.dec_msg[-1]))
        result = ''.join(self.dec_msg)
        self.output___txt_fld.insert(INSERT, result)
        self.cdbook___txt__fld.insert(INSERT, self.dec_codebook)


if __name__ == '__main__':
    root = Main(None)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

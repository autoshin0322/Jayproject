"""
    Hier wird der Lempel-Ziv Welch Algorithmus dargestellt. Program ist von Rosettacode:
        http://rosettacode.org/wiki/LZW_compression
    und das visuelle von uns.
"""

import sys
import subprocess
import Application
import pkg_resources
import matplotlib.pyplot as plt

from tkinter import Tk, Frame, Button, Label, Text, Scrollbar, VERTICAL, END, INSERT

required = {'matplotlib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


class Main(Tk):

    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.title('Lempel-Ziv-Welch')
        self.parent = parent

        self.buffer = []

        # ------------------------------------------------------------------------------------------------------------ #
        # --------------------------------- UNIVERSE ----------------------------------------------------------------- #
        self.universe = Frame(self)
        self.universe.pack()
        # ------------------------------------------------------------------------------------------------------------ #
        # --------------------------------- BUTTONS ------------------------------------------------------------------ #
        self.frame_btn = Frame(self.universe)
        self.frame_btn.grid(row=0, column=0)

        self.btn___return = Button(self.frame_btn, width=15, text='RETURN', relief='groove',
                                   command=self.return_)
        self.btn___encode = Button(self.frame_btn, width=15, text='COMPRESS', relief='groove',
                                   command=self.encode)
        self.btn___decode = Button(self.frame_btn, width=15, text='DECOMPRESS', relief='groove',
                                   command=self.decode)
        self.btn_validate = Button(self.frame_btn, width=15, text='STATS', relief='groove',
                                   command=self.stat)
        self.btn____clear = Button(self.frame_btn, width=15, text='CLEAR', relief='groove',
                                   command=self.clear)
        self.btn_____exit = Button(self.frame_btn, width=15, text='EXIT', relief='groove',
                                   command=self.destroy)
        self.btn___return.grid(row=0, column=0, padx=(0, 50))
        self.btn___encode.grid(row=0, column=1)
        self.btn___decode.grid(row=0, column=2)
        self.btn_validate.grid(row=0, column=3)
        self.btn____clear.grid(row=0, column=4)
        self.btn_____exit.grid(row=0, column=5, padx=(50, 0))
        # ------------------------------------------------------------------------------------------------------------ #
        # --------------------------------- TEXT FIELD --------------------------------------------------------------- #
        self.frame_txt = Frame(self.universe)
        self.frame_txt.grid(row=1, column=0)

        self.frame_input = Frame(self.frame_txt)
        self.frame_input.pack()

        self.lbl_input = Label(self.frame_input, text='INPUT', anchor='s')
        self.txt_field = Text(self.frame_input, width=96, height=7)
        self.scroll_bar_txt = Scrollbar(self.frame_input, command=self.txt_field.yview, orient=VERTICAL)
        self.txt_field['yscrollcommand'] = self.scroll_bar_txt.set
        self.txt_field.see(END)

        self.lbl_input.grid(row=0, column=0, padx=(10, 0), sticky='ew')
        self.txt_field.grid(row=1, column=0, padx=(10, 0), sticky='ew')
        self.scroll_bar_txt.grid(row=1, column=1, sticky='ns', padx=(0, 10))
        # ------------------------------------------------------------------------------------------------------------ #
        # --------------------------------- OUTPUTS ------------------------------------------------------------------ #
        self.outputs = Frame(self.universe)
        self.outputs.grid(row=2, column=0)

        self.frame_encoded_msg = Frame(self.outputs)
        self.frame_encoded_msg.grid(row=0, column=0)

        self.lbl_encoded_txt = Label(self.frame_encoded_msg, text='ENCODED MSG', relief='flat')
        self.lbl_encoded_fld = Text(self.frame_encoded_msg, state='disabled', width=96, height=7)
        self.scroll_bar_encode = Scrollbar(self.frame_encoded_msg, command=self.lbl_encoded_fld.yview,
                                           orient=VERTICAL)
        self.lbl_encoded_fld['yscrollcommand'] = self.scroll_bar_encode.set
        self.lbl_encoded_fld.see(END)

        self.lbl_encoded_txt.grid(row=0, column=0)
        self.lbl_encoded_fld.grid(row=1, column=0)
        self.scroll_bar_encode.grid(row=1, column=1, sticky='ns')

        self.frame_decoded_msg = Frame(self.outputs)
        self.frame_decoded_msg.grid(row=1, column=0, pady=(0, 20))

        self.lbl_decoded_txt = Label(self.frame_decoded_msg, text='DECODED MSG', relief='flat')
        self.lbl_decoded_fld = Text(self.frame_decoded_msg, state='disabled', width=96, height=7)
        self.scroll_bar_decode = Scrollbar(self.frame_decoded_msg, command=self.lbl_decoded_fld.yview,
                                           orient=VERTICAL)
        self.lbl_decoded_fld['yscrollcommand'] = self.scroll_bar_decode.set
        self.lbl_decoded_fld.see(END)

        self.lbl_decoded_txt.grid(row=0, column=0)
        self.lbl_decoded_fld.grid(row=1, column=0)
        self.scroll_bar_decode.grid(row=1, column=1, sticky='ns')

    @staticmethod
    def compress(uncompressed):
        """Compress a string to a list of output symbols."""

        # Build the dictionary.
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}

        w = ""
        result = []
        for c in uncompressed:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                # Add wc to the dictionary.
                dictionary[wc] = dict_size
                dict_size += 1
                w = c

        # Output the code for w.
        if w:
            result.append(dictionary[w])
        return result

    @staticmethod
    def decompress(compressed):
        """Decompress a list of output ks to a string."""
        from io import StringIO

        # Build the dictionary.
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}

        # use StringIO, otherwise this becomes O(N^2)
        # due to string concatenation in a loop
        result = StringIO()
        w = chr(compressed.pop(0))
        result.write(w)
        for k in compressed:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[0]
            else:
                raise ValueError('Bad compressed k: %s' % k)
            result.write(entry)

            # Add w+entry[0] to the dictionary.
            dictionary[dict_size] = w + entry[0]
            dict_size += 1

            w = entry
        return result.getvalue()

    def return_(self):
        """ Ruft das Auswahlmenü auf """
        self.update()
        Tk.destroy(self)

        main = Application.Main(None)
        main.eval('tk::PlaceWindow . center')
        main.mainloop()

    def clear(self):
        """ Löscht alle Einträge und leert alle Felder aus """
        self.txt_field.delete('1.0', END)

        self.lbl_encoded_fld.configure(state='normal')
        self.lbl_encoded_fld.delete('1.0', END)
        self.lbl_encoded_fld.configure(state='disabled')

        self.lbl_decoded_fld.configure(state='normal')
        self.lbl_decoded_fld.delete('1.0', END)
        self.lbl_decoded_fld.configure(state='disabled')

    def encode(self):
        """ Aktualiesiert das Textfeld für den encode """
        self.lbl_encoded_fld.configure(state='normal')
        self.lbl_encoded_fld.delete('1.0', END)
        self.lbl_encoded_fld.configure(state='disabled')

        string = self.txt_field.get('1.0', 'end-1c')
        encode = self.compress(string)
        tmp_enc = ''
        for i in encode:
            tmp_enc += str(i) + ', '
        str_res = tmp_enc[0:-2]
        self.lbl_encoded_fld.configure(state='normal')
        self.lbl_encoded_fld.insert(INSERT, str_res)
        self.lbl_encoded_fld.see(END)
        self.lbl_encoded_fld.configure(state='disabled')

    def decode(self):
        """ Aktualiesiert das Textfeld für den decode """
        self.lbl_decoded_fld.configure(state='normal')
        self.lbl_decoded_fld.delete('1.0', END)
        self.lbl_decoded_fld.configure(state='disabled')
        decode = self.lbl_encoded_fld.get('1.0', 'end-1c').split(", ")
        for i in range(len(decode)):
            decode[i] = int(decode[i])
        self.lbl_decoded_fld.configure(state='normal')
        self.lbl_decoded_fld.insert(INSERT, self.decompress(decode))
        self.lbl_decoded_fld.see(END)
        self.lbl_decoded_fld.configure(state='disabled')
        for i in decode:
            self.buffer.append(i)

    def stat(self):
        """ Hier wird das Ersparnis berechnet """
        compressed = self.buffer
        o_str_bit = len(self.txt_field.get('1.0', 'end-1c')) * 8
        temp = 8
        for i in range(len(compressed)):
            if int(compressed[i]) < 256:
                temp += 8
            elif int(compressed[i] < 512):
                temp += 9
            elif int(compressed[i] < 1024):
                temp += 10
            elif int(compressed[i] < 2048):
                temp += 11
            elif int(compressed[i] < 4096):
                temp += 12
            elif int(compressed[i] < 8192):
                temp += 13
            elif int(compressed[i] < 16384):
                temp += 14
            elif int(compressed[i] < 32768):
                temp += 15
            else:
                temp += 16
        temp_p = round(temp / o_str_bit, 2) * 100
        temp_q = 100 - temp_p

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Verwendete Bits', 'Ersparnis'
        sizes = [temp_p, temp_q]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.text(0, 1.1, 'Alle Bits: {}\nVewendete Bits: {}\nErsparte Bits: {}'.format(o_str_bit,
                                                                                       temp,
                                                                                       (o_str_bit-temp)))

        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()


if __name__ == "__main__":
    root = Main(None)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

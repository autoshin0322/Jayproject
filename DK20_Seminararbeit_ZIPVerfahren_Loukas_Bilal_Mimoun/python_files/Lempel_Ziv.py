"""
    Hier wird das Prinzip vom Lempel-Ziv Algorithmus visuell dargestellt.
"""

import sys
import time
import subprocess
import Decompress
import Application
import pkg_resources
import matplotlib.pyplot as plt

from tkinter import Tk, Frame, Button, Label, Text, Scrollbar, VERTICAL, END, INSERT, StringVar

required = {'matplotlib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


class Main(Tk):

    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.title('Lempel-Ziv - Visualised (principle)')
        self.parent = parent
        # ------------------------------------------------------------------------------------------------------------ #
        # --------------------------------- DECLARATIONS ------------------------------------------------------------- #
        self.bit_original = 0
        self.bit_wi_code = 0
        self.bit_wo_code = 0
        self.bit_temp = 0

        self.running = True  # ----- May be subject to change
        self.sender_codebook = []  # May be subject to change
        self.recver_codebook = []  # May be subject to change
        self.subsequence = []  # --- May be subject to change
        self.encoded_msg = []  # --- May be subject to change
        self.coded_msg = []  # ----- May be subject to change
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
        self.btn_compress = Button(self.frame_btn, width=15, text='COMPRESS', relief='groove',
                                   command=self.lempel_ziv)
        self.btn___decode = Button(self.frame_btn, width=15, text='DECOMPRESS', relief='groove',
                                   command=self.decode)
        self.btn_validate = Button(self.frame_btn, width=15, text='STATS', relief='groove',
                                   command=self.stat)
        self.btn____clear = Button(self.frame_btn, width=15, text='CLEAR', relief='groove',
                                   command=self.clear)
        self.btn_____exit = Button(self.frame_btn, width=15, text='EXIT', relief='groove',
                                   command=self.destroy)
        self.btn___return.grid(row=0, column=0, padx=(0, 50))
        self.btn_compress.grid(row=0, column=1)
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
        # --------------------------------- SENDER/RECEIVER ---------------------------------------------------------- #
        self.frame_sender_receiver = Frame(self.universe)
        self.frame_sender_receiver.grid(row=2, column=0)

        self.sender__display__txt = StringVar()
        self.sender__display__txt.set("Now being encoded\nN/A\n\nResult of the encoding\nN/A")
        self.lbl__sender__hdr = Label(self.frame_sender_receiver, text='SENDER')
        self.lbl__sender__txt = Label(self.frame_sender_receiver, textvariable=self.sender__display__txt,
                                      width=40, height=7, relief='groove')

        self.sent___display___txt = StringVar()
        self.sent___display___txt.set("sending\nN/A")
        self.lbl___sent___txt = Label(self.frame_sender_receiver, textvariable=self.sent___display___txt,
                                      width=30, height=7, relief='groove')

        self.receiver_display_txt = StringVar()
        self.receiver_display_txt.set("Now being decoded\nN/A\n\nResult of the decoding\nN/A")
        self.lbl_receiver_hdr = Label(self.frame_sender_receiver, text='RECEIVER')
        self.lbl_receiver_txt = Label(self.frame_sender_receiver, textvariable=self.receiver_display_txt,
                                      width=40, height=7, relief='groove')

        self.lbl__sender__hdr.grid(row=0, column=0, sticky='ew')
        self.lbl__sender__txt.grid(row=1, column=0, sticky='ew')

        self.lbl___sent___txt.grid(row=1, column=1, sticky='ew')

        self.lbl_receiver_hdr.grid(row=0, column=2, sticky='ew')
        self.lbl_receiver_txt.grid(row=1, column=2, sticky='ew')
        # ------------------------------------------------------------------------------------------------------------ #
        # -------------------------------- OUTPUTS ------------------------------------------------------------------- #
        self.outputs = Frame(self.universe)
        self.outputs.grid(row=3, column=0)

        self.frame_encoded_msg = Frame(self.outputs)
        self.frame_encoded_msg.grid(row=0, column=0, pady=(0, 20))

        self.lbl_encoded__txt = Label(self.frame_encoded_msg, text='ENCODED MSG', relief='flat')
        self.lbl_encoded__fld = Text(self.frame_encoded_msg, state='disabled', width=30, height=7)
        self.scroll_bar_encode = Scrollbar(self.frame_encoded_msg, command=self.lbl_encoded__fld.yview,
                                           orient=VERTICAL)
        self.lbl_encoded__fld['yscrollcommand'] = self.scroll_bar_encode.set
        self.lbl_encoded__fld.see(END)

        self.lbl_encoded__txt.grid(row=0, column=0, padx=(10, 0))
        self.lbl_encoded__fld.grid(row=1, column=0, padx=(10, 0))
        self.scroll_bar_encode.grid(row=1, column=1, sticky='ns')

        self.frame_codebook = Frame(self.outputs)
        self.frame_codebook.grid(row=0, column=1, pady=(0, 20))

        self.lbl_codebook_txt = Label(self.frame_codebook, text='DICTIONARY', relief='flat')
        self.lbl_codebook_fld = Text(self.frame_codebook, state='disabled', width=31, height=7)
        self.scroll_bar_codebook = Scrollbar(self.frame_codebook, command=self.lbl_codebook_fld.yview,
                                             orient=VERTICAL)
        self.lbl_codebook_fld['yscrollcommand'] = self.scroll_bar_codebook.set
        self.lbl_codebook_fld.see(END)

        self.lbl_codebook_txt.grid(row=0, column=0)
        self.lbl_codebook_fld.grid(row=1, column=0)
        self.scroll_bar_codebook.grid(row=1, column=1, sticky='ns')

        self.frame_decoded_msg = Frame(self.outputs)
        self.frame_decoded_msg.grid(row=0, column=2, pady=(0, 20))

        self.lbl_decoded__txt = Label(self.frame_decoded_msg, text='DECODED MSG', relief='flat')
        self.lbl_decoded__fld = Text(self.frame_decoded_msg, state='disabled', width=30, height=7)
        self.scroll_bar_decode = Scrollbar(self.frame_decoded_msg, command=self.lbl_decoded__fld.yview,
                                           orient=VERTICAL)
        self.lbl_decoded__fld['yscrollcommand'] = self.scroll_bar_decode.set
        self.lbl_decoded__fld.see(END)

        self.lbl_decoded__txt.grid(row=0, column=0)
        self.lbl_decoded__fld.grid(row=1, column=0)
        self.scroll_bar_decode.grid(row=1, column=1, sticky='ns', padx=(0, 10))

    @staticmethod
    def decode():
        """ Ruft die Klasse Decompress auf """
        dec = Decompress.Main(None)
        dec.mainloop()

    def display_text(self):
        """ Display Text """
        self.sender__display__txt.set("Now being encoded\nN/A\n\nResult of the encoding\nN/A")
        self.sent___display___txt.set("sending\nN/A")
        self.receiver_display_txt.set("Now being decoded\nN/A\n\nResult of the decoding\nN/A")

    def return_(self):
        """ Ruft das Auswahlmenü auf """
        self.update()
        Tk.destroy(self)

        main = Application.Main(None)
        main.eval('tk::PlaceWindow . center')
        main.mainloop()

    def clear(self):
        """ Löscht alle Einträge und leert alle Felder aus """
        self.sender_codebook.clear()  # May be subject to change
        self.recver_codebook.clear()  # May be subject to change
        self.subsequence.clear()  # --- May be subject to change
        self.encoded_msg.clear()  # --- May be subject to change
        self.coded_msg.clear()  # ----- May be subject to change

        self.bit_temp = 0
        self.bit_wo_code = 0
        self.bit_wi_code = 0
        self.bit_original = 0

        self.txt_field.delete('1.0', END)
        self.sender__display__txt.set('')
        self.sent___display___txt.set('')
        self.receiver_display_txt.set('')
        self.display_text()
        self.lbl_encoded__fld.configure(state='normal')
        self.lbl_codebook_fld.configure(state='normal')
        self.lbl_decoded__fld.configure(state='normal')
        self.lbl_encoded__fld.delete('1.0', END)
        self.lbl_codebook_fld.delete('1.0', END)
        self.lbl_decoded__fld.delete('1.0', END)
        self.lbl_encoded__fld.configure(state='disabled')
        self.lbl_codebook_fld.configure(state='disabled')
        self.lbl_decoded__fld.configure(state='disabled')

    def update_sent(self, text):
        """ Aktualiesiert das Label sent """
        self.sent___display___txt.set(text)

    def update_sender(self, text):
        """ Aktualiesiert das Label für den Sender """
        self.sender__display__txt.set(text)

        tmp_str = ''
        for i in self.sender_codebook:
            tmp_str += str(i) + '\n'
        self.lbl_codebook_fld.configure(state='normal')
        self.lbl_codebook_fld.delete('1.0', END)
        self.lbl_codebook_fld.tag_config('center', justify='center')
        self.lbl_codebook_fld.insert(INSERT, tmp_str)
        self.lbl_codebook_fld.tag_add('center', 1.0, 'end')
        self.lbl_codebook_fld.see(END)
        self.lbl_codebook_fld.configure(state='disabled')

    def compress(self):
        """ Wir komprimieren die Eingabe nach dem LZ Algorithmus """
        var = True
        counter = len(self.subsequence)
        for i in range(len(self.sender_codebook)):
            # Überprüfen ob der Teilstring sich im Codebook befindet.
            if self.sender_codebook[i][1] == self.subsequence[-1]:
                var = False
                break
        if var:
            self.sender_codebook.append((counter, self.subsequence[-1]))
        if len(self.subsequence[-1]) == 1:
            # ab hier wird komprimmiert
            self.bit_wo_code += 8
            self.coded_msg.append(self.subsequence[-1])
        else:
            if len(self.subsequence[-1]) == 2:
                temp = self.subsequence[-1][0]
            else:
                temp = self.subsequence[-1][0:-1]
            for i in range(len(self.sender_codebook)):
                if self.sender_codebook[i][1] == temp:
                    code = str(self.sender_codebook[i][0]) + self.subsequence[-1][-1]
                    self.coded_msg.append(code)
                    '''
                    if int(code[0:-1]) > 255:
                        self.bit_wo_code += 24
                    else:
                        self.bit_wo_code += 16
                    '''
                    # Bits werden gezählt
                    if int(code[0:-1]) < 256:
                        self.bit_wo_code += 8 + 8
                    elif int(code[0:-1]) < 512:
                        self.bit_wo_code += 9 + 8
                    elif int(code[0:-1]) < 1024:
                        self.bit_wo_code += 10 + 8
                    elif int(code[0:-1]) < 2048:
                        self.bit_wo_code += 11 + 8
                    elif int(code[0:-1]) < 4096:
                        self.bit_wo_code += 12 + 8
                    elif int(code[0:-1]) < 8192:
                        self.bit_wo_code += 13 + 8
                    elif int(code[0:-1]) < 16384:
                        self.bit_wo_code += 14 + 8
                    elif int(code[0:-1]) < 32768:
                        self.bit_wo_code += 15 + 8
                    else:
                        self.bit_wo_code += 16
        # Text der zu sehen ist / Ausgabe
        line_1 = 'Now being encoded\n'
        line_2 = '{}'.format(self.subsequence[-1])
        line_3 = 'Result of the encoding\n'
        line_4 = '{}'.format(self.coded_msg[-1])
        updated_text = line_1 + line_2 + '\n\n' + line_3 + line_4

        line_5 = 'sending\n'
        line_6 = '{}'.format(self.coded_msg[-1])
        text = line_5 + line_6

        self.update_sent(text)
        self.update_sender(updated_text)

    def update_receiver(self, text):
        """ Aktualiesiert das Label für den Empfänger """
        self.receiver_display_txt.set(text)

        temp_cod = '['
        for i in range(len(self.coded_msg)):
            if i == len(self.coded_msg) - 1:
                temp_cod += "'" + str(self.coded_msg[i]) + "'" + ']'
            else:
                temp_cod += "'" + str(self.coded_msg[i]) + "'" + ', '
        display_encoded = temp_cod
        self.lbl_encoded__fld.configure(state='normal')
        self.lbl_encoded__fld.delete('1.0', END)
        self.lbl_encoded__fld.insert(INSERT, display_encoded)
        self.lbl_encoded__fld.see(END)
        self.lbl_encoded__fld.configure(state='disabled')

        temp_dec = ""
        for i in self.encoded_msg:
            temp_dec += i
        display_decoded = temp_dec
        self.lbl_decoded__fld.configure(state='normal')
        self.lbl_decoded__fld.delete('1.0', END)
        self.lbl_decoded__fld.insert(INSERT, display_decoded)
        self.lbl_decoded__fld.see(END)
        self.lbl_decoded__fld.configure(state='disabled')

    def decompress(self):
        """ Wir dekomprimieren die Eingabe nach dem LZ Algorithmus """
        var = True
        temp_1 = self.coded_msg[-1]
        if len(temp_1) == 1:
            # Wenn die zu dekondierende Teilnachricht die länge 1 hat zb nur a => wird in encoded_msg abgespeichert
            self.encoded_msg.append(temp_1)
            for j in range(len(self.recver_codebook)):
                # und wenn sich das Symbol nicht im recver_codebook befindet, speicher wir das in recver_codebook(Z 232)
                if self.recver_codebook[j][1] == temp_1:
                    var = False
                    break
            if var:
                self.recver_codebook.append((len(self.coded_msg), temp_1))
        else:
            # ab hier werden alle Teilnachrichten ab länge=2 dekodiert.
            if len(temp_1) == 2:
                temp = temp_1[0]
            else:
                temp = temp_1[0:-1]
            for i in range(len(self.recver_codebook)):
                if self.recver_codebook[i][0] == int(temp):
                    code = str(self.recver_codebook[i][1]) + temp_1[-1]
                    for j in range(len(self.recver_codebook)):
                        if self.recver_codebook[j][1] == code:
                            var = False
                            break
                    if var:
                        self.recver_codebook.append((len(self.coded_msg), code))
                    var = True
                    self.encoded_msg.append(code)
        # Ausgabe
        line_1 = 'Now being decoded\n'
        line_2 = '{}'.format(temp_1)
        line_3 = 'Result of the decoding\n'
        line_4 = '{}'.format(self.encoded_msg[-1])
        updated_text = line_1 + line_2 + '\n\n' + line_3 + line_4

        self.update_receiver(updated_text)
        # Setzt man variabel ein, wie lange das Program warten soll (Sekunden)
        time.sleep(0.01)

    def stat(self):
        """ Hier wird das Ersparnis berechnet """
        self.bit_temp = round(self.bit_wo_code / self.bit_original, 2)

        temp_p = 100 * round(self.bit_wo_code / self.bit_original, 2)
        temp_q = 100 - temp_p

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Verwendete Bits', 'Ersparnis'
        sizes = [temp_p, temp_q]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=180)
        ax1.text(0, 1.1, 'Alle Bits: {}\n'
                         'Vewendete Bits: {}\n'
                         'Ersparte Bits: {}'.format(self.bit_original,
                                                    self.bit_wo_code,
                                                    (self.bit_original - self.bit_wo_code)))
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

    def lempel_ziv(self):
        """ Hier werden Teilstrings erstellt und die kodier/dekodier Aufrufe gestartet """
        if self.running:
            string = self.txt_field.get('1.0', 'end-1c')

            self.bit_original = len(string) * 8

            cnt_i = 0
            tmp_sub = ""
            while True:
                self.update()
                if cnt_i < len(string):
                    tmp_sub += string[cnt_i]
                    if self.subsequence.__contains__(tmp_sub):
                        if cnt_i + 1 == len(string):
                            self.subsequence.append(tmp_sub)
                            self.compress()
                            self.decompress()
                            break
                        else:
                            cnt_i += 1
                    else:
                        self.subsequence.append(tmp_sub)
                        tmp_sub = ""
                        cnt_i += 1
                        self.compress()
                        self.decompress()
                else:
                    break


if __name__ == "__main__":
    root = Main(None)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

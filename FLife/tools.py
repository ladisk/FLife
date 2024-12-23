from matplotlib.pyplot import table
import numpy as np
from scipy import stats
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time

def relative_error(value, value_ref):
    """Return the relative error 

    :param value: [int,float]
        Value to compare
    :param value: [int,float]
        Reference value
    :return err: float
        Relative error.
    """ 
    return (value-value_ref)/value_ref

def basquin_to_sn(Sf, b, range = False):
    """Converts Basquin equation parameters Sf and b to fatigue life parameters C and k,
    as defined in [2]. Basic form of Basquin equation is used here: Sa = Sf* (2*N)**k 

    :param Sf: [int,float]
        Fatigue strength coefficient [MPa**k].
    :param b : [int,float]
        Fatigue strength exponent [/]. Represents S-N curve slope.
    :param range : bool
        False/True sets returned value C with regards to amplitude / range count, respectively.
    :return C, k: float
        C - S-N curve intercept [MPa**k], k - S-N curve inverse slope [/].
    """ 
    if not range:
        k = -1/b 
        C = 0.5*Sf**k
        return C,k

    else: 
        k = -1/b 
        C = 0.5*(2*Sf)**k
        return C,k

def pdf_rayleigh_sum(m0L, m0H):
    """Returns PDF (Probability Density Function) of sum of 2 Rayleigh distributed RV (Random Variables).

    :param m0L, moH: [int,float]
        Variance of first and second Rayleigh distributed RV. 
    :return pdf: function
    """
    m0 = m0L + m0H
    m0L_norm = m0L/m0
    m0H_norm = m0H/m0

    def pdf(x):
        pdf = 1 / np.sqrt(m0) * (m0L_norm * x/np.sqrt(m0) * np.exp(- (x/np.sqrt(m0))**2 /(2*m0L_norm)) \
            + m0H_norm * x/np.sqrt(m0) * np.exp(- (x/np.sqrt(m0))**2 /(2*m0H_norm)) \
            + np.sqrt(2*np.pi*m0L_norm*m0H_norm) * ((x/np.sqrt(m0))**2 -1) * np.exp(-(x/np.sqrt(m0))**2/2) \
            * (stats.norm.cdf(np.sqrt(m0L_norm/m0H_norm)*x/np.sqrt(m0)) + stats.norm.cdf(np.sqrt(m0H_norm/m0L_norm)*x/np.sqrt(m0)) - 1))
        return pdf
    return pdf


def random_gaussian(freq, PSD, T, fs, rg=None, random_amplitude=False, **kwargs):
    """
    Stationary Gaussian realization of random process, characterized by PSD.
    
    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Area under PSD curve represents variance of random process.
    
    :param freq: array
        Frequency vector
    :param PSD: array
        One-sided power spectral density [unit^2].
    :param T: int,float
        Length of returned random process [s].
    :param fs: int,float
        Sampling frequency [Hz].
    :param rg: numpy.random._generator.Generator
        Initialized Generator object
    :param random_amplitude: Boolean
        If true, Rayleigh distributed amplitude is used in addition 
        to uniformly distributed phase. Defaults to False
    :return: t, signal
        Time and stationary Gaussian realization of random process
    
    Notes
    -----
    PSD data points are interpolated using numpy.interp() function. Option for 
    interpolation function will be added in the future.
    
    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis.
    Dover Publications, 2005
    """ 
    # time and frequency data
    if np.__version__>='2.0.0':
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    var = trapezoid(PSD,freq)
    N = int(T * fs)
    M = N//2 + 1
    t = np.arange(0,N) / fs # time vector

    M = N//2 + 1
    freq_new = np.arange(0, M, 1) / N  * fs # frequency vector
    PSD_new = np.interp(freq_new, freq, PSD, left=0, right=0)
    
    if rg == None:
        rg = np.random.default_rng()
    
    if isinstance(rg, np.random._generator.Generator):
        
        phase = rg.uniform(0, 1, len(PSD_new))
        
        ampl_spectra = np.sqrt(PSD_new * N * fs / 2)  # amplitude spectra modulus
        if random_amplitude == True:
            ampl_spectra = np.array([rg.rayleigh(size=1, scale=x)[0] for x in ampl_spectra]) # Rayleigh distributed amplitudes

        ampl_spectra_random = ampl_spectra * np.exp(
            1j * phase * 2 * np.pi)  # amplitude spectra,  random phase

    else:
        raise ValueError(
            '`rg` must be initialized Generator object (numpy.random._generator.Generator)!'
        )

    signal = np.fft.irfft(ampl_spectra_random, n=N)  # time signal
    signal = signal/np.std(signal) * var**.5
    return t, signal


class PSDgen(object):
    """
    Get Power Spectral Density (PSD) via graph or table.

    PSD is determined by linear interpolation of selected points, in linear 
    or logarithmic scale. 
    """
    def __init__(self):
        self.data_entries = {} # dictionary for point data entries
        self.scale = 'lin' # 'lin' - linear, 'log' - logarithmic
        self.deltaF_lock = False # disable change of freq resolution after firts point selection
        self.shift_is_held = False
        self.D_is_held = False
        self._button_press_data = False 
        self._data_point_drag = False
        
        # Tkinter
        self.root = tk.Tk()
        self.root.title("FLife: PSD input")
        self.root.bind_all("<1>", lambda event:event.widget.focus_set()) #Set focus on widget when cklicked, left
        self.root.bind_all("<2>", lambda event:event.widget.focus_set()) #Set focus on widget when cklicked, midddle
        self.root.bind_all("<3>", lambda event:event.widget.focus_set()) #Set focus on widget when cklicked, right

        # Tkinter layout; grid() geometry manager. Top level grid is one row and three columns
        # Grid resizing settings
        self.root.rowconfigure(0, minsize=400, weight=1) #height of first row, i.e. window height
        self.root.columnconfigure(0, minsize=80, weight=0) #width of first column
        self.root.columnconfigure(1, minsize=700, weight=1) #width of second column
        self.root.columnconfigure(2, minsize=100, weight=0) #width of third column

        # First column; menu: help, export and confirm buttons
        frm_menu = tk.Frame(self.root)
        # Help button
        btn_help = tk.Button(frm_menu, text="Help", command=self.help)
        btn_help.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        # Export PSD button
        btn_export = tk.Button(frm_menu, text="Export PSD", command=self.save_file)
        btn_export.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        # Add frame frm_menu to grid
        frm_menu.grid(row=0, column=0, sticky="nsew")

        # Second column; frame for help text, canvas, toolbar and scale
        frm_PSD_display = tk.Frame(self.root)
        # Frame resizing settings
        frm_PSD_display.grid_rowconfigure(1, weight=1) # Canvas
        frm_PSD_display.grid_columnconfigure(0, weight=1)
        # Help
        self.lbl_help = tk.Label(frm_PSD_display)
        self.lbl_help.grid(row=0, column=0, sticky='nsew', pady = 5)
        self.text_initial = 'SHIFT+LMB click to add point, SHIFT+RMB click&drag to move point, SHIFT+D+LMB click to remove data point.'
        self.lbl_help['text'] = self.text_initial
        #Figure and Canvas widget
        self.fig = Figure(figsize=(7, 4))
        self.freq_max = 10000 # [Hz]
        self.PSD_max = 100 # [Unit^2/H]
        self.dfreq = 1.0 # [Hz], default value
        self.freq = np.arange(0, self.freq_max, self.dfreq)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, which='both')
        self.ax.set_xlim([0,self.freq_max])
        self.ax.set_ylim([0,self.PSD_max])
  
        # Integrate matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm_PSD_display)  
        canvas_widget = self.canvas.get_tk_widget()
        # Add canvas to frame
        canvas_widget.grid(row=1, column=0, sticky="nsew")
      	# Connecting functions to event manager
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.select_data_point)
        self.fig.canvas.mpl_connect('button_release_event', self.move_data_point)
        self.fig.canvas.mpl_connect('motion_notify_event', self.drag_data_point)

        # Toolbar  widget
        toolbar = NavigationToolbar2Tk(self.canvas, frm_PSD_display, pack_toolbar=False)
        # Add toolbar to frame
        toolbar.grid(row=2, column=0, sticky='nsw')

        #Scale buttons are contained in frame frm_scale
        frm_scale = tk.Frame(frm_PSD_display)
        btn_scale_lin = tk.Button(frm_scale, text="lin", command=self.set_scale_lin)
        btn_scale_log = tk.Button(frm_scale, text="log", command=self.set_scale_log)
        btn_scale_lin.grid(row=0, column=1, sticky="snew", padx=2, pady=7)
        btn_scale_log.grid(row=0, column=2, sticky="snew", padx=2, pady=7)
        # Add frm_scale to frm_PSD_display
        frm_scale.grid(row=2, column=0, sticky="sne")
        # Add frame frm_PSD_display to grid
        frm_PSD_display.grid(row=0, column=1, sticky="nsew")

        # Third column; frame for frequency resolution button, PSD table and Confirm PSD Button
        frm_column3 = tk.Frame(self.root)
        frm_column3.grid_rowconfigure(1, weight=1) # configure table size settings
        # Frequency resolution button frame
        # label
        frm_deltaf = tk.Frame(frm_column3)
        lbl_deltaF = tk.Label(frm_deltaf, text='df [Hz]', width=5, relief=tk.FLAT)
        lbl_deltaF.grid(row=0, column=0, sticky='nsw', pady=8)
        #Entry
        self.ent_deltaF = tk.Entry(frm_deltaf, width=10)
        self.ent_deltaF.insert(0, str(self.dfreq))
        self.ent_deltaF.grid(row=0, column=1, sticky = 'nsw', pady=8, padx=2)
        #button
        btn_deltaF = tk.Button(frm_deltaf, text='\N{RIGHTWARDS BLACK ARROW}', command=self.confirm_deltaf)
        btn_deltaF.grid(row=0, column=2, sticky='nsw', pady=8, padx=2)
        #add frm frm_deltaF to frm_column3
        frm_deltaf.grid(row=0, column=0)
        # Frame for PSD table
        self.frm_PSD_table = tk.Frame(frm_column3)
        # 0.th row: column name and units
        lbl_nmbr = tk.Label(self.frm_PSD_table, width=3, relief=tk.FLAT, text='#')
        lbl_nmbr.grid(row=0, column=0, sticky='nswe', padx=1, pady=4)
        lbl_freq = tk.Label(self.frm_PSD_table, width=5, relief=tk.FLAT, text='f [Hz]')
        lbl_freq.grid(row=0, column=1, sticky='nswe', padx=1, pady=4)
        lbl_PSD = tk.Label(self.frm_PSD_table, width=12, relief=tk.FLAT, text='PSD [Unit^2/Hz]')
        lbl_PSD.grid(row=0, column=2, sticky='nswe', padx=1, pady=4)
        # 1.st row is empty at initialization
        # data-point number
        lbl_nmbr = tk.Label(self.frm_PSD_table, width=1, relief=tk.SUNKEN)
        lbl_nmbr['text']='1.'
        lbl_nmbr.grid(row=1, column=0, sticky='nswe', padx=1, pady=1)
        #frequency
        num_freq  = tk.StringVar()
        ent_freq = tk.Entry(self.frm_PSD_table, textvariable=num_freq, width=10)
        ent_freq.grid(row=1, column=1, sticky='nswe', padx=0, pady=1)
        #PSD value
        num_PSD  = tk.StringVar()
        ent_PSD= tk.Entry(self.frm_PSD_table, textvariable=num_PSD, width=10)
        ent_PSD.grid(row=1, column=2, sticky='nswe', padx=0, pady=1)
        #save to variable
        self.new_entry =  [None, None, lbl_nmbr, ent_freq, ent_PSD] # use for new entry via table
        #confirm button
        self.enter_PSD = tk.Button(self.frm_PSD_table, text="Confirm table")
        self.enter_PSD.bind('<Button-1>', self.add_data_from_table)
        # Add confirm buttor to frm_PSD_table
        self.enter_PSD.grid(row=2, column=2, sticky='nse', padx=1, pady=1)
        # Add frame frm_PSD_table to frm_column3
        self.frm_PSD_table.grid(row=1, column=0, sticky='news')
        # Confirm PSD button
        btn_confirm = tk.Button(frm_column3, height = 1, text="Confirm PSD and exit", command=self.on_closing)
        btn_confirm.grid(row=2, column=0, sticky="nsew", padx=5, pady=7)
        # Add frm_column3 to grid
        frm_column3.grid(row=0, column=2, sticky='nsew', )

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def help(self):
        window = tk.Toplevel(self.root)

        # Title
        window.title("FLife: PSD input - Help")
        lbl_help = tk.Label(window, width=70, height=10)
        help_text = [
            " Set frequency resolution df with click on right arrow button before selecting data points.\
            \n\nAdd a point with SHIFT + LEFT mouse button.\
            \nTranslate a point with SHIFT + RIGHT mouse button hold.\
            \nDelete a point with SHIFT + D + LEFT mouse button.\
            \n\nPoint can be entered (or changed) via PSD table with click on `Confirm table'.\
            \n\nData point selection is activated by mouse button click on graph area."
        ]
        lbl_help['text'] = ''.join(help_text)
        lbl_help.pack(padx=10, pady=10)

    def save_file(self):
        """
        Save the current PSD as a new file.
        """
        filepath = asksaveasfilename(
            defaultextension="txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        # If data points exist
        try:
            PSD = self.get_PSD()
            # Save to txt
            np.savetxt(filepath, PSD, fmt=['%d','%d'])
        except:
            pass
        self.root.title(f"FLife: PSD input - {filepath}")

    def confirm_deltaf(self):
        """
        Set frequency resolution for PSD. Defaults to 1 Hz.
        """
        #unlock change in freq resolution
        if len(self.data_entries) == 0:
            self.dfreq = float(self.ent_deltaF.get())
            self.ent_deltaF.delete(0, tk.END)
            self.ent_deltaF.insert(0, f'{self.dfreq:.1f}')
            self.freq = np.arange(0, self.freq_max, self.dfreq)
        else:
            self.ent_deltaF.config(background="RED")
            self.root.update()
            time.sleep(0.1)
            self.ent_deltaF.delete(0, tk.END)
            self.ent_deltaF.insert(0, f'{self.dfreq:.1f}')
            self.ent_deltaF.config(background="WHITE")            

    def get_PSD(self):
        """
        Return PSD, defined by linear interpolation of entered data points.
        """
        # If data points exist
        try:
            # Unpack data points
            list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = self._list_unpack_data_entries()
            freq_data = np.array(list_freq)
            psd_data =  np.array(list_psd)
            # Interpolate data to frequency vector, from first to last frequency in data points
            freq_min_index = np.where(self.freq == freq_data[0])[0][0]
            freq_max_index = np.where(self.freq == freq_data[-1])[0][0]
            psd_values = np.interp(self.freq[freq_min_index:freq_max_index], freq_data, psd_data)
            PSD = np.column_stack([self.freq[freq_min_index:freq_max_index], psd_values])
        except:
            # Return zero valued PSD
            psd_values = np.zeros_like(self.freq)
            PSD = np.column_stack([self.freq, psd_values])
        return PSD

    def on_closing(self):
        self.root.destroy()

    def on_key_press(self, event):
        """
        Function triggered on key press (SHIFT, D).
        """
        # Change flag
        if event.key == 'shift':
            self.shift_is_held = True
        elif event.key == 'd':
            self.D_is_held = True
        elif event.key == 'D':
            self.D_is_held = True
        self.interactive_help()

    def on_key_release(self, event):
        """
        Function triggered on key release (SHIFT, D).
        """
        if event.key == 'shift':
            self.shift_is_held = False
        elif event.key == 'd':
            self.D_is_held = False
        elif event.key == 'D':  
            self.D_is_held = False
        self.interactive_help()

    def interactive_help(self):
        # Change lbl_help text
        if self.shift_is_held == True and self.D_is_held == False:
            self.lbl_help['text'] = 'LMB click to add data point, RMB click&drag to move data point, D+LMB click to remove data point.'
        elif self.shift_is_held == True and self.D_is_held == True:
            self.lbl_help['text'] = 'LMB click to delete data point.'
        elif self.shift_is_held == False and self.D_is_held == True:
            self.lbl_help['text'] = 'SHIFT+LMB click to remove data point.'
        elif self.shift_is_held == False and self.D_is_held == False:
            self.lbl_help['text'] = self.text_initial

    def set_scale_lin(self):
        self.scale= 'lin'
        self.plot_PSD()
        
    def set_scale_log(self):
        self.scale= 'log'
        self.plot_PSD()
    
    def plot_PSD(self):
        """
        Plot PSD points, linearly interpolated.
        """
        # get current axes limit
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
            
        try: # if at least one data point exist, unpack data_entries
            list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = self._list_unpack_data_entries() 
        except: # no data
            list_freq, list_psd = [], []

        # Plot
        self.ax.clear()
        if self.scale == 'lin':
            self.ax.plot(list_freq, list_psd, 'ko-', ms=3)
        elif self.scale == 'log':
            self.ax.loglog(list_freq, list_psd, 'ko-', ms=3)

        self.ax.grid(True, which='both')
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.fig.canvas.draw()

    def _unpack_data_entries(self):
        """
        Sorts and unpack self.data_entries. Returned data are sorted by increasing frequency.
        """
        sorted_input = sorted(self.data_entries.items(), key=lambda item: item[1][0]) #sorted freqeuncy; data_entries.values[0]
        tuples = zip(*sorted_input)
        data_entries_nmbr, data_entries = [list(tuple) for tuple in  tuples] #split consecutive input number and data
        return data_entries

    def _list_unpack_data_entries(self):
        """
        Returneds list of data, sorted by increasing frequency.
        """
        data_entries_data = self._unpack_data_entries()
        list_data_entries = [list(data) for data in zip(*data_entries_data)]
        return list_data_entries
        
    def select_data_point(self, event):
        """
        Select data point (freqeuncy, value) from graph.
        """
        try:  #if mouse is over graph
            y_data = float(event.ydata)
            x_data = float(event.xdata)

            if event.button == 1 and self.shift_is_held and not self.D_is_held:
                self.add_data_from_graph(x_data,y_data)
                self.refresh_PSD_table()
                self.plot_PSD()
            elif event.button == 1 and self.shift_is_held and self.D_is_held:
                self.remove_data(x_data)
                self.refresh_PSD_table()
                self.plot_PSD()
            elif event.button == 3 and self.shift_is_held:
                self._button_press_data = event  # save data point to be translated
        except:
            pass

    def move_data_point(self, event):
        """
        Moves selected data point to another location on graph. Frequency and PSD value are updated, accordingly.
        Function is triggered by release of RIGHT mouse button.
        """
        try:  #if mouse is over Figure object
            y_data = float(event.ydata)
            x_data = float(event.xdata)
            
            if event.button == 3 and self.shift_is_held and self._button_press_data:
                self._drag_data_point(x_data, y_data, flag = False) #False flag disables moving_data_point() 
                self._button_press_data = False
                # replot PSD
                self.plot_PSD()
                # refresh Table
                self.refresh_PSD_table()
        except:
            pass

    def _drag_event_handle(self, event):
        """
        Inner function for use with drag_data_point(). When drag_data_point() is running, motion_notify_event is funneled to 
        this function.
        """
        if self._drag_data_point == False:
            #self.root.update()
            # rebind event back to func drag_data_point
            self.fig.canvas.mpl_connect('motion_notify_event', self.drag_data_point)
        else:
            pass

    def drag_data_point(self, event):
        """
        Updates data point while mouse is moving and RIGHT mouse button is held.
        """
        if self._data_point_drag == True:
            # rebind event to func _drag_event_handle when drag event is in progress
            self.fig.canvas.mpl_connect('motion_notify_event', self._drag_event_handle)
            pass
        else:
            try:  #if mouse is over Figure object
                y_data = float(event.ydata)
                x_data = float(event.xdata)

                if event.button == 3 and self.shift_is_held and self._button_press_data:
                    self._data_point_drag = True
                    self._drag_data_point(x_data, y_data, flag = event)
                    self.root.update()
                    # set flag
                    self._data_point_drag = False
                    # replot PSD
                    self.plot_PSD()
                    # refresh Table
                    self.refresh_PSD_table()
            except:
                pass

    def _drag_data_point(self, x_data, y_data, flag):
        """
        Subfunction used in functions move_data_point and drag_data_point. 
        """
        x_last = self._button_press_data.xdata

        # Unpack data_entries
        list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = self._list_unpack_data_entries()

        # Get old frequency 
        x_ind = np.abs(x_last - self.freq).argmin()
        freq_old = self.freq[x_ind]

        #new freqeuncy
        x_ind = np.abs(x_data - self.freq).argmin()
        freq_new = self.freq[x_ind]

        if freq_old in list_freq:
            indx = list_freq.index(freq_old)
            list_freq[indx] = freq_new
            list_psd[indx] = y_data

            data_entries = dict([(indx,u) for indx,u in enumerate(zip(list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD))])
            self.data_entries = dict(sorted(data_entries.items(), key=lambda item: item[1][0])) #sorted freqeuncy; data_entries.values[0]

        self._button_press_data = flag

    def remove_data(self, x_data):
        # Get selected frequency 
        list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = self._list_unpack_data_entries()
        freq_sel = self._data_select_zoom(x_data, list_freq, 0.05)

        for_deletion = None
        for key in self.data_entries.keys(): 
            if freq_sel == self.data_entries[key][0]:
                for_deletion = key
                
        if for_deletion !=  None:
            deleted = self.data_entries.pop(for_deletion)
            # remove data point from table
            lbl_nmbr = deleted[2]
            lbl_nmbr.grid_remove()
            ent_freq = deleted[3]    
            ent_freq.grid_remove() 
            ent_PSD = deleted[4]    
            ent_PSD.grid_remove()

    def _data_select_zoom(self, x_data, list_freq, zoom):
            """
            Select closest frequency.
            """
            freq_sel_index = np.abs(x_data - self.freq).argmin()
            freq_sel = self.freq[freq_sel_index]
            index_min, index_max = self.ax.get_xlim()

            # Frequency index range
            delta_index = index_max - index_min
            zoom_range = int(zoom * delta_index)
            if zoom_range%2 == 0:
                pass
            else:
                zoom_range += 1
            freq_zoom_indexes = np.arange(freq_sel_index-int(zoom_range/2), freq_sel_index+int(zoom_range/2)+1, 1)
            # 
            zoomed_frequencies = []
            for freq in list_freq:
                if freq in self.freq[freq_zoom_indexes]:
                    delta = np.abs(freq_sel - freq)
                    zoomed_frequencies.append((freq, delta))
            #sort to closest frequency
            zoomed_frequencies = sorted(zoomed_frequencies, key=lambda x: x[1])

            try:
                freq = zoomed_frequencies[0][0]
            except:
                freq = freq_sel
            return freq

    def _sort_data_entries(self):
        """
        Function sort and updates data entries. Sorting is done according to frequency, keys are updated.
        """
        list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = self._list_unpack_data_entries()
        # update keys, in case of deleted data point
        data_entries = dict([(indx,u) for indx,u in enumerate(zip(list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD))])
        self.data_entries = dict(sorted(data_entries.items(), key=lambda item: item[1][0])) #sorted frequency; data_entries.values[0]

    def add_data_from_table(self, event):
        """
        Add data from table to self.data_entries
        """
        # Read table and update data entries
        table_len = len(self.data_entries) 
        # Sort data
        if table_len > 0:
            self._sort_data_entries()

        index = 0 #if table_len = 0, index=0
        for index in range(table_len): 
            freq, PSD, lbl_nmb, ent_freq, ent_PSD = self.data_entries[index]
            freq = float(ent_freq.get())
            PSD = float(ent_PSD.get())
            self.data_entries[index] = [freq, PSD, lbl_nmb, ent_freq, ent_PSD] 
        
        try:
            if float(self.new_entry[3].get()) >= 0 and float(self.new_entry[4].get()) >= 0: # if new data point is given
                # read entered data point and add to data entries
                new_ent_freq = self.new_entry[3]
                new_ent_PSD = self.new_entry[4]
                freq_new = float(new_ent_freq.get())
                PSD_new = float(new_ent_PSD.get())
                self._add_data(freq_new, PSD_new)
                # Clear new entry text
                new_ent_freq.delete(0, 'end')
                new_ent_PSD.delete(0, 'end')
        except:
            pass

        # Refresh Table
        self.refresh_PSD_table()
        # Replot PSD
        self.plot_PSD()
        
    def add_data_from_graph(self, x_data, y_data):
        """
        Adds data from clik on graph.
        """
        self._add_data(x_data, y_data)

    def _add_data(self, freq_input, psd_input):
        """
        Read input data and adds to dict self.data_entries
        """
        try: # if at least one data point exist, unpack data_entries
            list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = self._list_unpack_data_entries() 
        except: # else create empyt lists
            list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD = [list() for u in range(5)] 

        # Get selected frequency 
        # if bigger then max freq
        if freq_input > self.freq[-1]:
            self.freq = np.arange(0, freq_input + self.dfreq, self.dfreq)
            self.freq_max = self.freq[-1]

        x_ind = np.abs(freq_input - self.freq).argmin()
        freq_sel = self.freq[x_ind]
        
        # Check axes limit
        if freq_sel >= self.freq_max*0.95:
            self.freq_max = freq_sel * 1.1
            self.freq = np.arange(0, self.freq_max, self.dfreq)
        if psd_input >= self.PSD_max*0.95:
            self.PSD_max = psd_input*1.1

        # update PSD_data; if freq_sel already in PSD-data, replace, otherwise append
        if freq_sel in list_freq:
            pass # if data point with freq_sel already exist, it should be dragged to another point on graph
        else:
            list_freq.append(freq_sel) 
            list_psd.append(psd_input)
            # Add widgets
            row_indx = len(list_freq) + 1
            # data-point number
            lbl_nmbr = tk.Label(self.frm_PSD_table, width=1, relief=tk.SUNKEN)
            lbl_nmbr['text']= str(row_indx)
            lbl_nmbr.grid(row=row_indx, column=0, sticky='nswe', padx=1, pady=1)
            list_lbl_nmbr.append(lbl_nmbr)
            # frequency
            num_freq  = tk.StringVar()
            ent_freq = tk.Entry(self.frm_PSD_table, textvariable=num_freq, width=10)
            ent_freq.grid(row=row_indx, column=1, sticky='nswe', padx=0, pady=1)
            list_ent_freq.append(ent_freq)
            # PSD value
            num_PSD  = tk.StringVar()
            ent_PSD= tk.Entry(self.frm_PSD_table, textvariable=num_PSD, width=10)
            ent_PSD.grid(row=row_indx, column=2, sticky='nswe', padx=0, pady=1)
            list_ent_PSD.append(ent_PSD)
            # confirm button
            self.enter_PSD.grid(row=row_indx+1, column=2, sticky='nse', padx=1, pady=1)
            # sort and save
            data_entries = dict([(indx,u) for indx,u in enumerate(zip(list_freq, list_psd, list_lbl_nmbr, list_ent_freq, list_ent_PSD))])
            self.data_entries = dict(sorted(data_entries.items(), key=lambda item: item[1][0])) # sorted frequency; data_entries.values[0]

    def refresh_PSD_table(self):
        """Updates PSD table, based on self.data_entries
        """
        try:
            # Unpack data_entries
            data_entries = self._unpack_data_entries()

            #update widgets
            for row_indx, entry_data in enumerate(data_entries):
                row_indx += 1 #zero-th row is used for column description
                #freq and PSD value
                freq = f'{entry_data[0]:.0f}'
                PSD = f'{entry_data[1]:.1f}'
                #Input point number
                lbl_nmbr = entry_data[2]
                lbl_nmbr['text']= str(row_indx)
                lbl_nmbr.grid(row=row_indx, column=0, sticky='nswe', padx=1, pady=1)
                #Freq
                ent_freq = entry_data[3]
                ent_freq.delete(0, tk.END)
                ent_freq.insert(0, freq)
                ent_freq.grid(row=row_indx, column=1, sticky='nswe', padx=0, pady=1)
                #PSD
                ent_PSD = entry_data[4]
                ent_PSD.delete(0, tk.END)
                ent_PSD.insert(0, PSD)
                ent_PSD.grid(row=row_indx, column=2, sticky='nswe', padx=0, pady=1)
        
            #Change row of entry line
            self.new_entry[2]['text'] = str(row_indx+1)
            self.new_entry[2].grid(row=row_indx+1, column=0, sticky='nswe', padx=0, pady=1)
            self.new_entry[3].grid(row=row_indx+1, column=1, sticky='nswe', padx=0, pady=1)
            self.new_entry[4].grid(row=row_indx+1, column=2, sticky='nswe', padx=0, pady=1)
        
        except:
            self.new_entry[2]['text'] = str(1)
            self.new_entry[2].grid(row=1, column=0, sticky='nswe', padx=0, pady=1)
            self.new_entry[3].grid(row=1, column=1, sticky='nswe', padx=0, pady=1)
            self.new_entry[4].grid(row=1, column=2, sticky='nswe', padx=0, pady=1)

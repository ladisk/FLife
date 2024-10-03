import pyvista as pv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def set_mesh(self,filename):
  
    self.mesh = pv.read(filename)


def pick_point(self):
    
    # Create the plotter object
    p = pv.Plotter()
    f = Figure()
    canvas = FigureCanvas(f)
    ax = f.add_subplot(111)
    f.tight_layout(pad=2)

    def refresh_psd_graph():
        ax.clear()
        if hasattr(self, 'psd'):
            ax.semilogy(self.psd[:, 0], self.psd[:, 1])
            ax.axvline(self.psd[freq_index, 0], color='C1')
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('PSD [Unit^2/Hz]')
        f.tight_layout()

        # Redraw the Matplotlib figure
        
        f.canvas.draw()

    # Initial scalar index (start in the middle of the frequency range)
    freq_index = len(self.eq_psd_multipoint[0][0]) // 2
    self.mesh['eq_stress'] = self.eq_psd_multipoint[0][:, freq_index]  # Initial scalars

    # Add your mesh and keep a reference to the actor

    actor = p.add_mesh(
        self.mesh,
        scalars='eq_stress',
        pickable=True,
        show_scalar_bar=True,
        scalar_bar_args={
            
            'position_x': 0.5,
            'position_y': 0.1,
            'width': 0.45
        }
    )

    custom_camera = pv.Camera()
    
    # Set the custom camera
    p.camera = custom_camera

    # Define a callback function for the point picker
    def point_picker_callback(point):
        nonlocal freq_index  # Ensure freq_index is accessible
        if point is not None:
            # Find the index of the picked point in the mesh
            point_id = self.mesh.find_closest_point(point)
            self.point_id = point_id
            # Update the eq_stress scalars
            self.set_eq_stress(self.eq_psd_multipoint[0][point_id], self.eq_psd_multipoint[1])
            refresh_psd_graph()
            p.render()

    # Define a callback function for the slider
    def slider_callback(value):
        nonlocal freq_index  # Access the freq_index from the outer scope
        freq_index = int(round(value))  # Round the slider value to the nearest whole number
        # Directly modify the scalars of the mesh in-place
        self.mesh['eq_stress'] = self.eq_psd_multipoint[0][:, freq_index]
        refresh_psd_graph()
        vmax = 0.75*np.max(self.mesh['eq_stress'])
        # Update the scalar range on the actor's mapper
        actor.mapper.scalar_range = (0, vmax)
        p.render()

    # Enable point picking with the callback function
    p.enable_point_picking(callback=point_picker_callback, pickable_window=False,show_message='Right click to select point')

    # Add a slider to control the scalar index dynamically
    slider_widget = p.add_slider_widget(
        slider_callback,         # Function to call when the slider moves
        rng=[0, self.eq_psd_multipoint[0].shape[1] - 1],  # Range of the slider
        value=freq_index,      # Default value
        title="Frequency",       # Title of the slider
        pointa=(.005, .1),        # Starting point of the slider in the plotter window
        pointb=(.45, .1),        # Ending point of the slider in the plotter window
        style='modern',          
    )

    slider_rep = slider_widget.GetRepresentation()

    # Hide the numerical display above the slider
    slider_rep.SetShowSliderLabel(False)

    # Add the Matplotlib chart to the plotter
    
    h_chart = pv.ChartMPL(f, size=(0.45, 0.5), loc=(0.005, 0.2))
    h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
    p.add_chart(h_chart)

    # Show the plotter
    p.show(full_screen=True)



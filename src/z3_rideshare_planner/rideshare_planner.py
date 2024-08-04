import carshare
import pandas as pd
import googlemaps

class RidesharePlanner:
    def __init__(self, API_KEY):
        self.gmaps_client = googlemaps.Client(key=API_KEY)
        self.API_KEY = API_KEY

    def configure(self, passenger_name_addr, driver_name_addr, destination_):
        carshare.setup(passenger_name_addr, driver_name_addr, destination_, self.API_KEY)
        self.destination = destination_
        self.passenger_df = pd.DataFrame(passenger_name_addr, columns=['person', 'address'])
        self.driver_df = pd.DataFrame(driver_name_addr, columns=['person', 'address'])

    def solve(self, n_seats):
        model = carshare.search_opt(n_seats)
        paths, max_time = carshare.parse_plan(model)
        return RidesharePlanner.RidePlan(paths, max_time, self.passenger_df, self.driver_df, self.destination, self.API_KEY)

    class RidePlan:
        def __init__(self, paths, time_bound, passenger_df, driver_df, destination, KEY):
            self.paths = paths
            self.time_bound = time_bound
            self.passenger_df = passenger_df
            self.driver_df = driver_df
            self.destination = destination
            self.KEY = KEY

        def __str__(self):
            output = ''
            for d in range(len(self.paths)):
                s = f"{self.driver_df['person'].iloc[d]}: {self.driver_df['person'].iloc[d]}"
                for passenger in self.paths[d]:
                    if passenger != self.passenger_df.shape[0]:
                        s += f" -> {self.passenger_df['person'].iloc[passenger]}"
                    else:
                        s += f' -> destination'
                output += s + '\n'
            output += f'Time bound: {self.time_bound}'
            return output
        
        def visualize(self, output_mode='html', relative_path=None, colormap='Set1'):
            # if output_mode == 'html': store the output as html file, relative_path required
            # if output_mode == 'display': return the Folium figure object
            if output_mode == 'html' and relative_path is None:
                raise ValueError('relative_path must be provided when output_mode is html')
            figure = carshare.draw_plan(self.paths, googlemaps.Client(key=self.KEY), colormap=colormap, output_mode=output_mode)
            if output_mode == 'html':
                figure.save(relative_path)
                return 
            else:
                return figure



import googlemaps
import pandas as pd
import sys
from z3 import *
from datetime import datetime
import googlemaps
import gmaps
import gmaps.datasets
import folium
import polyline as pl
import numpy as np
import matplotlib.pyplot as plt


# Main #

def main(passenger_name_addr, driver_name_addr, destination_, n_seats, API_KEY):
    global KEY, passengers, drivers, destination, n_p, n_d, passenger_addresses, driver_addresses
    global distances, passenger_df, driver_df, people_df, t_list, T_list, N_list, gmaps_client, now
    
    #Initialization
    KEY = API_KEY
    passengers = passenger_name_addr
    drivers = driver_name_addr
    destination = destination_
    n_p = len(passengers)
    n_d = len(drivers)
    passenger_addresses = [p[1] for p in passengers]
    driver_addresses = [d[1] for d in drivers]
    distances = construct_distance_matrix(passenger_addresses + driver_addresses + [destination], KEY)
    passenger_df = pd.DataFrame(passengers, columns=['person', 'address'])
    driver_df = pd.DataFrame(drivers, columns=['person', 'address'])
    people_df = pd.concat([passenger_df, driver_df]).reset_index().drop(columns=['index'])

    # Variable declaration
    t_list = [Bool('t_%s' % i) for i in range(n_d * (n_p + 1) ** 2)]
    T_list = [Int('T_%s' % i) for i in range(n_d * (n_p + 1))]
    N_list = [Int('N_%s' % i) for i in range(n_d * (n_p + 1))]

    # Generate plan
    model = search_opt(n_seats)
    print_plan(*parse_plan(model))

    # Save plan to html
    gmaps.configure(api_key=KEY)  # Fill in with your API key
    gmaps_client = googlemaps.Client(key=KEY)  # Fill in with your API key
    now = datetime.now()
    figure = draw_plan(model)
    figure.save('map.html')

    

#---------------------#

# Distance Matrix functions #

def get_distance_matrix(origins, destinations, API_KEY):
    # Google Map distanceMatrix API can support up to 11 addresses at one time
    gmaps_client = googlemaps.Client(key=API_KEY)

    distances = pd.DataFrame(index=pd.MultiIndex.from_product([origins, destinations], names=['Origin', 'Destination']), columns=['duration'])

    rows = gmaps_client.distance_matrix(origins, destinations)['rows']
    row_idx = 0
    for row in rows:
        elements = row['elements']
        col_idx = 0
        for element in elements:
            if element['status'] == 'OK':
                distances.loc[(origins[row_idx], destinations[col_idx]), 'duration'] = element['duration']['value']
            else:
                print(f'DISTANCE NOT FOUND: {origins[row_idx]} - {destinations[col_idx]}')
                sys.exit()
            col_idx += 1
        row_idx += 1
    return distances

def split_addresses(addrs, maxlen=10):
    l = len(addrs)
    if l <= maxlen:
        return [addrs]
    n = l // maxlen
    output = []
    for i in range(n):
        output.append(addrs[i * maxlen: (i+1) * maxlen])
    output.append(addrs[n * maxlen:])
    return output

def construct_distance_matrix(addrs, API_KEY, maxlen=10):
    addrs_splitted = split_addresses(addrs, maxlen)
    out = pd.DataFrame(
        index=pd.MultiIndex.from_product([[], []], names=['Origin', 'Destination']),
        columns=['duration']
    )
    for sublist1 in addrs_splitted:
        for sublist2 in addrs_splitted:
            out = pd.concat([out, get_distance_matrix(sublist1, sublist2, API_KEY)], ignore_index=False)
    return out

#---------------------#

# Z3 functions #

# Map to variable
def t(i, j, k):
    if not (i >= 0 and i < n_d and j >= -1 and j <= n_p and k >= 0 and k <= n_p):
        print(f'Error: Index t({i}, {j}, {k}) not valid')
        return
    return (t_list[i * (n_p + 1) ** 2 + j * (n_p + 1) + k] if j != -1 else t(i, n_p, k))

def T(i, j):
    if not (i >= 0 and i < n_d and j >= 0 and j <= n_p):
        print(f'Error: Index T({i}, {j}) not valid')
        return
    return T_list[i * n_d + j]

def N(i, j):
    if not (i >= 0 and i < n_d and j >= 0 and j <= n_p):
        print(f'Error: Index N({i}, {j}) not valid')
        return
    return N_list[i * n_d + j]

#---------------------#

# Proposition Construction #

def drives(driver, passenger):
    if passenger == -1:
        return Bool(True)
    return Or(*tuple(
        [t(driver, prevPassenger, passenger) for prevPassenger in range(-1, n_p) if prevPassenger != passenger]
        ))


def T_update(driver, prevPassenger, passenger):
    if prevPassenger == -1:
        if passenger == n_p:
            d = distances.loc[(people_df.iloc[driver + n_p]['address'], destination)]['duration']
        else:
            d = distances.loc[(people_df.iloc[driver + n_p]['address'], people_df.iloc[passenger]['address'])]['duration']
        return T(driver, passenger) == d
    else:
        if passenger == n_p:
            d = distances.loc[(people_df.iloc[prevPassenger]['address'], destination)]['duration']
        else:
            d = distances.loc[(people_df.iloc[prevPassenger]['address'], people_df.iloc[passenger]['address'])]['duration']
        return T(driver, passenger) == T(driver, prevPassenger) + d
    
    
def N_update(driver, prevPassenger, passenger):
    if passenger == n_p and prevPassenger == -1:
        return N(driver, passenger) == 1
    if prevPassenger == -1:
        return N(driver, passenger) == 2
    if passenger == n_p:
        return N(driver, passenger) == N(driver, prevPassenger)
    return N(driver, passenger) == N(driver, prevPassenger) + 1


def D(driver, prevPassenger, passenger):
    conjunctive_list = []
    conjunctive_list.append(t(driver, prevPassenger, passenger))

    for alt_d in range(n_d):
        for alt_p in range(-1, n_p):
            if alt_p != prevPassenger or alt_d != driver:
                conjunctive_list.append(Not(t(alt_d, alt_p, passenger)))
        for alt_p in range(0, n_p + 1):
            if prevPassenger != -1 and (alt_p != passenger or alt_d != driver):
                conjunctive_list.append(Not(t(alt_d, prevPassenger, alt_p)))
            elif prevPassenger == -1 and alt_d == driver and alt_p != passenger:
                conjunctive_list.append(Not(t(alt_d, prevPassenger, alt_p)))
    
    conjunctive_list.append(drives(driver, prevPassenger))
    
    conjunctive_list = list(set(conjunctive_list))
    
    return And(*tuple(conjunctive_list), T_update(driver, prevPassenger, passenger), N_update(driver, prevPassenger, passenger))


def driverGuarantee(driver, seats):
    disjunctive_list = [
        And(
            t(driver, prevPassenger, n_p),
            *tuple(
                [Not(t(driver, alt_p, n_p)) for alt_p in range(-1, n_p) if alt_p != prevPassenger]
            ),
            drives(driver, prevPassenger),
            T_update(driver, prevPassenger, n_p),
            N_update(driver, prevPassenger, n_p)
        )
        for prevPassenger in range(-1, n_p)
    ]
    disjunctive_list = list(set(disjunctive_list))
    return And(Or(*tuple(disjunctive_list)), N(driver, n_p) <= seats)

    
def cost_limit(cost):
    return And(
        *tuple(
            [(T(driver, n_p) < cost) for driver in range(n_d)]
        )
    )


def passengerGuarantee(passenger):
    disjunctives = []
    for driver in range(n_d):
        for prevPassenger in range(-1, n_p):
            if prevPassenger != passenger:
                disjunctives.append(D(driver, prevPassenger, passenger))
    return Or(*tuple(disjunctives))

def generate_plan(cost, n_seats, ignore_cost=False):
    s = Solver()

    if isinstance(n_seats, int) or isinstance(n_seats, float):
        if n_seats * n_d < n_d + n_p:
            print('Error: Seats insufficient')
            return (False, None)
        for driver in range(n_d):
            s.add(driverGuarantee(driver, n_seats))
    elif (isinstance(n_seats, tuple) or isinstance(n_seats, list)) and len(n_seats) == n_d:
        if sum(n_seats) * n_d < n_d + n_p:
            print('Error: Seats insufficient')
            return (False, None)
        for driver, n in enumerate(n_seats):
            s.add(driverGuarantee(driver, n))
    else:
        print('Error: n_seats format wrong')
        return (False, None)
        
    for passenger in range(n_p):
        s.add(passengerGuarantee(passenger))
        
    if s.check() == sat:
        if ignore_cost:
            # print('SUCCESS: Found one plan ignoring the cost')
            return (True, s.model())
    else:
        print('Not Satisfied')
        return (False, None)
    s.add(cost_limit(cost))
    if s.check() == sat:
        # print('SUCCESS')
        return (True, s.model())
    else:
        # print('No plan under the cost')
        return (False, None)
    
#---------------------#
    
# Find and parse optimal solution #

def search_opt(n_seats):
    attempt = generate_plan(0, n_seats, True)
    if attempt[0]:
        model = attempt[1]
        time = max([model.eval(T(i, n_p)).as_long() for i in range(n_d)])
    else:
        return None
    while time > 0:
        attempt = generate_plan(time - 1, n_seats)
        if not attempt[0]:
            break
        model = attempt[1]
        time = max([model.eval(T(i, n_p)).as_long() for i in range(n_d)])
    return model

def reverse_t(i):
    driver = i // ((n_p + 1) ** 2)
    i = i % ((n_p + 1) ** 2)
    prevPassenger = i // (n_p + 1)
    i = i % (n_p + 1)
    if prevPassenger == n_p:
        prevPassenger = -1
    return (driver, prevPassenger, i)

def reverse_T(T_val):
    return (T_val % n_d, T_val // n_d)

def parse_plan(model):
    def backtrace(driver, passenger):
        if passenger == -1:
            return []
        prev = next(filter(lambda e: e[0] == driver and e[2] == passenger, edges))[1]
        return backtrace(driver, prev) + [passenger]

    t_indices = [i for i in range(len(t_list)) if is_true(model.eval(t_list[i]))]
    max_time = max([model.eval(T(i, n_p)).as_long() for i in range(n_d)])
    
    edges = [reverse_t(i) for i in t_indices]
    paths = [backtrace(driver, n_p) for driver in range(n_d)]
    return (paths, max_time)

def print_plan(paths, max_time):
    for d in range(len(paths)):
        s = f"{driver_df['person'].iloc[d]}: {driver_df['person'].iloc[d]}"
        for passenger in paths[d]:
            if passenger != n_p:
                s += f" -> {passenger_df['person'].iloc[passenger]}"
            else:
                s += f' -> destination'
        print(s)
    print(f'Time bound: {max_time}')

#---------------------#

# Visualization functions #

def draw_path(map, path, driver, color):
    def draw_section(start, end):
        polyline = gmaps_client.directions(start, end, mode="driving", departure_time=now)[0]['overview_polyline']['points']
        points = pl.decode(polyline)
        folium.PolyLine(points, color= f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", weight=4, opacity=1).add_to(map)
        return
    
    driver_location = get_lat_lon(driver_df['address'].iloc[driver])
    draw_section(driver_location, get_lat_lon(people_df['address'].iloc[path[0]]))
    add_marker(map, driver_location, 0, driver_df['person'].iloc[driver] + ' (driver)', color)
    for i in range(len(path)):
        start = get_lat_lon(people_df['address'].iloc[path[i]])
        if i == len(path) - 1:
            end = get_lat_lon(destination)
        else:
            end = get_lat_lon(people_df['address'].iloc[path[i + 1]])
        draw_section(start, end)
        add_marker(map, start, i + 1, people_df['person'].iloc[path[i]], color)

    return

def add_marker(map, location, index, label, color):
    icon_html = lambda d: f'''
        <div style="background-color: rgb({color[0]},{color[1]},{color[2]}); border-radius: 50%; 
            width: 30px; height: 30px; text-align: center; line-height: 30px; 
            color: {'gray' if index == 0 else 'white'}; font-size: 14pt;">
            {d}
        </div>
    '''
    folium.Marker(location=location,
            popup=folium.Popup(label, max_width=70),
            icon=folium.DivIcon(html=icon_html(index))).add_to(map)
    return
    
def get_lat_lon(address):
    result = gmaps_client.geocode(address)
    if len(result) == 0:
        print(f'Error: Address {address} not found')
        return
    loc = result[0]['geometry']['location']
    return (loc['lat'], loc['lng'])

def draw_plan(model, colormap='Set1'):
    # Initialize the map
    all_addrs = passenger_addresses + driver_addresses + [destination]
    all_addrs = np.array([get_lat_lon(addr) for addr in all_addrs])
    southwest, northeast =  all_addrs.min(axis=0), all_addrs.max(axis=0)
    difference  = northeast - southwest
    southwest -= difference * 0.2
    northeast += difference * 0.2
    bounds = [southwest.tolist(), northeast.tolist()]
    figure = folium.Figure(width=1000, height=500)
    map = folium.Map(location=all_addrs.mean(axis=0), zoom_start=12, min_zoom = 11, max_bounds=True,
                     min_lat=bounds[0][0], max_lat=bounds[1][0], min_lon=bounds[0][1], 
                     max_lon=bounds[1][1]).add_to(figure)
    
    
    paths, max_time = parse_plan(model)
    paths = [path[:-1] for path in paths] # Remove destination from all paths

    # Figure the colors
    colors = list(plt.colormaps[colormap].colors)
    colors = [tuple(int(255 * c) for c in color) for color in colors]
    if len(paths) + 1 > len(colors):
        colors += [tuple(np.random.randint(0, 256, 3)) for _ in range(len(paths) + 1 - len(colors))]
    else:
        colors = colors[:len(paths) + 1]

    for i in range(len(paths)):
        draw_path(map, paths[i], i, colors[i]) # Draw path for driver i
    add_marker(map, get_lat_lon(destination), 'D', destination, colors[-1])
    # print(f'Time bound: {max_time}')
    return figure




KEY = 'AIzaSyDt76ZoGVu1Dtd2UMQ7Hwd8RktdlWbHN2o'
passengers = [
    ('A1', '8775 Costa Verde Blvd, San Diego, CA'),
    ('B1', '8800 Lombard Pl, San Diego, CA'),
    ('C1', '9600 Campus Point Dr, La Jolla, CA'),
    ('D1', '7655 Palmilla Dr, San Diego, CA'),
    ('E1', '8636 Villa La Jolla Dr, La Jolla, CA'),
    ('F1', '7867 Camino Aguila, San Diego, CA'),
    ('G1', '10374 Wateridge Cir, San Diego, CA'),
]

drivers = [
    ('A2', '9500 Gilman Dr, La Jolla, CA'),
    ('B2', '9515 Genesee Ave, San Diego, CA'),
    ('C2', '1415 Orion Dr, San Diego, CA')
]

destination = '3435 Del Mar Heights Rd, San Diego, CA'

n_p = len(passengers)
n_d = len(drivers)
passenger_addresses = [p[1] for p in passengers]
driver_addresses = [d[1] for d in drivers]
distances = construct_distance_matrix(passenger_addresses + driver_addresses + [destination], KEY)

passenger_df = pd.DataFrame(passengers, columns=['person', 'address'])
driver_df = pd.DataFrame(drivers, columns=['person', 'address'])
people_df = pd.concat([passenger_df, driver_df]).reset_index().drop(columns=['index'])


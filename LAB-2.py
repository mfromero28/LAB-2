import math
from typing import List, Dict, Tuple, Set
import pandas as pd
import folium
import webbrowser
import os

#Bueno, les voy a comentar esto para que se entienda
class Graph:

    def __init__(self, n: int, directed: bool = False):
        self.n = n
        self.directed = directed
        self.L: List[List[Tuple[int, float]]] = [[] for _ in range(n)]  # esta es la lista de adyacencia (nodo, peso)
        self.airport_data: Dict[int, Dict] = {}  # no sé si esta vuelta es así pero aqui se pone la info de cada airport por indice
    
    def add_airport_info(self, idx: int, info: Dict):
       #aqui se hace de lo que estaba hablando antes de la info de los airport
        self.airport_data[idx] = info

    #esto basicamente añade una arista entre dos nodos en el grafo
    def add_edge(self, u: int, v: int, lat_u: float, lon_u: float, lat_v: float, lon_v: float) -> bool:
        if 0 <= u < self.n and 0 <= v < self.n:
            weight = haversine(lat_u, lon_u, lat_v, lon_v)
            self.L[u].append((v, weight))
            if not self.directed:
                self.L[v].append((u, weight))
            return True
        return False
#Bueno aqui se hace por Kruskal el arbol del expansion minima, pero puse las siglas en ingles xd
    def find_MST(self):
        # Lista de todas las aristas
        edges = []
        for u in range(self.n):
            for v, weight in self.L[u]:
                if u < v:  # esto es para evitar duplicados 
                    edges.append((weight, u, v))

        # Ordenamos las aristas por peso
        edges.sort()

        # Implementamos el algoritmo de Kruskal
        parent = list(range(self.n)) 
        rank = [0] * self.n

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
#esto es un locuron para actualizar el rango, si no se puede hacer lo cambio y ya
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1

        # el MST
        mst_weight = 0
        mst_edges = []
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst_weight += weight
                mst_edges.append((u, v, weight))

        return mst_weight, mst_edges

#esto es para ver si es conexo o no y saber la cantidad de componentes
    def is_connected(self) -> Tuple[bool, List[Set[int]]]:
        visited = [False] * self.n
        components = []

        def dfs(u, component):
            visited[u] = True
            component.add(u)
            for v, _ in self.L[u]:
                if not visited[v]:
                    dfs(v, component)

        for u in range(self.n):
            if not visited[u]:
                component = set()
                dfs(u, component)
                components.append(component)

        is_connected = len(components) == 1
        return is_connected, components

    def shortest_path(self, start: int) -> Dict[int, Tuple[float, List[int]]]:
        #aqui vamos a implementar Dijkstra machi, eduardo dijo que no librerias que hicieran esto, usé heapq pero,
        # eso es para optimizar la cola de prioridad en el algoritmo de Dijkstra :) digo yo que se vale
        import heapq
        dist = {i: float('inf') for i in range(self.n)}
        dist[start] = 0
        prev = {i: None for i in range(self.n)}
        pq = [(0, start)]  # (distancia acumulada, nodo)

        while pq:
            current_dist, u = heapq.heappop(pq)
            if current_dist > dist[u]:
                continue

            for v, weight in self.L[u]:
                new_dist = current_dist + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Reconstrucción de los caminos más cortos
        paths = {}
        for v in range(self.n):
            if dist[v] < float('inf'):
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = prev[node]
                paths[v] = (dist[v], path[::-1])

        return paths

    def display_airport_info(self, idx: int):
        #esto es breve, mostrar la info en cuestion, de un airport
        if idx in self.airport_data:
            info = self.airport_data[idx]
            print(f"Aeropuerto: {info['code']} - {info['name']}")
            print(f"Ciudad: {info['city']}, País: {info['country']}")
            print(f"Coordenadas: ({info['latitude']}, {info['longitude']})")
        else:
            print("Aeropuerto no encontrado.")

    def create_map(self):
        # Crear un mapa centrado en las coordenadas medias
        avg_lat = sum(info['latitude'] for info in self.airport_data.values()) / self.n
        avg_lon = sum(info['longitude'] for info in self.airport_data.values()) / self.n
        mapa = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)
        
        # Agregar marcadores para cada aeropuerto
        for idx, info in self.airport_data.items():
            folium.Marker(
                location=[info['latitude'], info['longitude']],
                popup=f"{info['code']} - {info['name']}, {info['city']}, {info['country']}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa)
        
        # Guardar el mapa en un archivo HTML
        mapa.save("mapa_aeropuertos.html")
        print("El mapa se ha guardado como 'mapa_aeropuertos.html'.")
    def show_shortest_path_on_map(self, start: int, end: int):
        # Mostrar el camino mínimo entre dos aeropuertos en el mapa
        paths = self.shortest_path(start)
        if end not in paths:
            print("No hay camino disponible entre estos aeropuertos.")
            return

        path_info = paths[end]
        path = path_info[1]

        mapa = folium.Map(location=[self.airport_data[start]['latitude'], self.airport_data[start]['longitude']], zoom_start=5)

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            folium.Marker(
                location=[self.airport_data[u]['latitude'], self.airport_data[u]['longitude']],
                popup=f"{self.airport_data[u]['code']} - {self.airport_data[u]['name']}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa)
            folium.Marker(
                location=[self.airport_data[v]['latitude'], self.airport_data[v]['longitude']],
                popup=f"{self.airport_data[v]['code']} - {self.airport_data[v]['name']}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa)
            folium.PolyLine(
                locations=[(self.airport_data[u]['latitude'], self.airport_data[u]['longitude']),
                           (self.airport_data[v]['latitude'], self.airport_data[v]['longitude'])],
                color="green"
            ).add_to(mapa)

        mapa.save("camino_minimo.html")
        print("El camino mínimo se ha guardado como 'camino_minimo.html'.")
        print(f"Distancia total: {path_info[0]:.2f} km")
        webbrowser.open("camino_minimo.html")

        

# Haversine para el cálculo de distancias entre dos coordenadas geográficas(es rarito que pero si se puede usar)
#grafics en progreso...
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # distancia en km

# Agregar una función para mostrar el camino mínimo sobre el mapa
def show_shortest_path_on_map(self, start: int, end: int):
        paths = self.shortest_path(start)
        if end not in paths:
            print("No hay camino disponible entre estos aeropuertos.")
            return
        
        path_info = paths[end]
        path = path_info[1]

        # Crear un mapa centrado en el primer aeropuerto
        mapa = folium.Map(location=[self.airport_data[start]['latitude'], self.airport_data[start]['longitude']], zoom_start=5)
        
        # Agregar los marcadores y las líneas del camino
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            folium.Marker(
                location=[self.airport_data[u]['latitude'], self.airport_data[u]['longitude']],
                popup=f"{self.airport_data[u]['code']} - {self.airport_data[u]['name']}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa)
            folium.Marker(
                location=[self.airport_data[v]['latitude'], self.airport_data[v]['longitude']],
                popup=f"{self.airport_data[v]['code']} - {self.airport_data[v]['name']}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa)
            folium.PolyLine(
                locations=[(self.airport_data[u]['latitude'], self.airport_data[u]['longitude']),
                           (self.airport_data[v]['latitude'], self.airport_data[v]['longitude'])],
                color="green"
            ).add_to(mapa)
        
        # Guardar el mapa en un archivo HTML
        mapa.save("camino_minimo.html")
        print("El camino mínimo se ha guardado como 'camino_minimo.html'.")
        print(f"Distancia total: {path_info[0]:.2f} km")

# Cargar el dataset de vuelos
df = pd.read_csv('flights_final.csv')

# Mapeo de códigos de aeropuertos a índices del grafo
airport_to_idx = {}
current_idx = 0

for index, row in df.iterrows():
    source_code = row['Source Airport Code']
    dest_code = row['Destination Airport Code']

    # Asignar índices a los aeropuertos si aún no lo tienen
    if source_code not in airport_to_idx:
        airport_to_idx[source_code] = current_idx
        current_idx += 1
    if dest_code not in airport_to_idx:
        airport_to_idx[dest_code] = current_idx
        current_idx += 1

# Crear el grafo con el número total de aeropuertos(aun no se muestra,ok?)
n_airports = len(airport_to_idx)
graph = Graph(n_airports)

# Agregar las aristas y la información de los aeropuertos al grafo
for index, row in df.iterrows():
    u = airport_to_idx[row['Source Airport Code']]
    v = airport_to_idx[row['Destination Airport Code']]
    lat_u, lon_u = row['Source Airport Latitude'], row['Source Airport Longitude']
    lat_v, lon_v = row['Destination Airport Latitude'], row['Destination Airport Longitude']
    
    graph.add_edge(u, v, lat_u, lon_u, lat_v, lon_v)

    # Agregar información del aeropuerto
    graph.add_airport_info(u, {
        'code': row['Source Airport Code'],
        'name': row['Source Airport Name'],
        'city': row['Source Airport City'],
        'country': row['Source Airport Country'],
        'latitude': lat_u,
        'longitude': lon_u
    })
    graph.add_airport_info(v, {
        'code': row['Destination Airport Code'],
        'name': row['Destination Airport Name'],
        'city': row['Destination Airport City'],
        'country': row['Destination Airport Country'],
        'latitude': lat_v,
        'longitude': lon_v
    })

#   AJA Y ESTO ES LO QUE SALE EN CONSOLA
while True:
    print("\nMenú de opciones:")
    print("1. Verificar si el grafo es conexo")
    print("2. Encontrar el árbol de expansión mínima")
    print("3. Mostrar caminos mínimos desde un aeropuerto")
    print("4. Mostrar información de un aeropuerto")
    print("5. Mostrar mapa de aeropuertos")
    print("6. Mostrar camino mínimo entre dos aeropuertos en el mapa")
    print("7. Salir")


    opcion = int(input("Seleccione una opción: "))

    if opcion == 1:
        connected, components = graph.is_connected()
        if connected:
            print("El grafo es conexo.")
        else:
            print(f"El grafo no es conexo. Tiene {len(components)} componentes.")
            for i, component in enumerate(components):
                print(f"Componente {i+1}: {len(component)} vértices")

    elif opcion == 2:
        connected, components = graph.is_connected()
        if connected:
            mst_weight, mst_edges = graph.find_MST()
            print(f"Peso total del árbol de expansión mínima: {mst_weight:.2f} km")
            print("Aristas del árbol de expansión mínima:")
            for u, v, weight in mst_edges:
                info_u = graph.airport_data[u]
                info_v = graph.airport_data[v]
                print(f"{info_u['code']} ({info_u['city']}) - {info_v['code']} ({info_v['city']}): {weight:.2f} km")
        else:
            print(f"El grafo no es conexo. Tiene {len(components)} componentes.")
            for i, component in enumerate(components):
                print(f"\nComponente {i+1} (con {len(component)} vértices):")
                subgraph = Graph(len(component))
                component_list = list(component)
                
                mapping = {old: new for new, old in enumerate(component_list)}
                
                # Agregar las aristas del subgrafo
                for u in component:
                    for v, weight in graph.L[u]:
                        if v in component:
                            subgraph.add_edge(mapping[u], mapping[v], 
                                            graph.airport_data[u]['latitude'], graph.airport_data[u]['longitude'],
                                            graph.airport_data[v]['latitude'], graph.airport_data[v]['longitude'])
                            subgraph.add_airport_info(mapping[u], graph.airport_data[u])
                            subgraph.add_airport_info(mapping[v], graph.airport_data[v])

                # Calcular el MST de la componente
                mst_weight, mst_edges = subgraph.find_MST()
                print(f"Peso total del MST de la componente {i+1}: {mst_weight:.2f} km")
                for u, v, weight in mst_edges:
                    info_u = subgraph.airport_data[u]
                    info_v = subgraph.airport_data[v]
                    print(f"{info_u['code']} ({info_u['city']}) - {info_v['code']} ({info_v['city']}): {weight:.2f} km")

    elif opcion == 3:
        code = input("Ingrese el código del aeropuerto: ")
        if code in airport_to_idx:
            start = airport_to_idx[code]
            paths = graph.shortest_path(start)
            # Mostrar los 10 caminos más largos
            top_10 = sorted(paths.items(), key=lambda x: x[1][0], reverse=True)[:10]
            for i, (v, (dist, path)) in enumerate(top_10, 1):
                print(f"{i}. {graph.airport_data[v]['code']} - {graph.airport_data[v]['name']}: {dist:.2f} km")
        else:
            print("Código de aeropuerto no encontrado.")

    elif opcion == 4:
        code = input("Ingrese el código del aeropuerto: ")
        if code in airport_to_idx:
            graph.display_airport_info(airport_to_idx[code])
        else:
            print("Código de aeropuerto no encontrado.")

    elif opcion == 5:
        graph.create_map()

    elif opcion == 6:
        start_code = input("Ingrese el código del aeropuerto de origen: ")
        end_code = input("Ingrese el código del aeropuerto de destino: ")
        if start_code in airport_to_idx and end_code in airport_to_idx:
            start = airport_to_idx[start_code]
            end = airport_to_idx[end_code]
            graph.show_shortest_path_on_map(start, end)
        else:
            print("Código de aeropuerto no encontrado.")
    
    elif opcion == 7:
        break

    else:
        print("Opción no válida, por favor intente nuevamente.")        
    
# tipo_green.py
# Allows one-shot determination of a website's TipoRed (type) based on its IP address, or
# generation of (semi) infinite random IP's of a given type.


import random

reserved_ips = {
    "0.0.0.0": "0.255.255.255",
    "10.0.0.0": "10.255.255.255",
    "100.64.0.0": "100.127.255.255",
    "127.0.0.0": "127.255.255.255",
    "169.254.0.0": "169.254.255.255",
    "172.16.0.0": "172.31.255.255",
    "192.0.0.0": "192.0.0.255",
    "192.0.2.0": "192.0.2.255",
    "192.88.99.0": "192.88.99.255",
    "198.18.0.0": "198.19.255.255",
    "198.51.100.0": "198.51.100.255",
    "203.0.113.0": "203.0.113.255",
    "224.0.0.0": "239.255.255.255",
    "240.0.0.0": "255.255.255.254",
    "255.255.255.255": "255.255.255.255",
}

TipoRed = [
    "Comisaria",
    "Universidades",
    "Supermercados",
    "FastFood",
    "Taller",
    "MobileShop",
    "Hospitales",
    "Bancos",
    "Particulares",
    "MailServices",
    "HackShop",
    "TiendaInformatica",
    "NetServices",
    "HardwareManufacturer",
    "Neurobox",
    "Tutorial",
    "CurrencyCreation",
]

seed = 0

def ip_to_int(ip):
    """Return the integer representation of an IP address"""
    array = ip.split(".")
    return (
        (int(array[0]) << 24)
        | (int(array[1]) << 16)
        | (int(array[2]) << 8)
        | int(array[3])
    )


def int_to_ip(ip_int):
    """Convert an integer to an IP address"""
    return f"{(ip_int >> 24) & 255}.{(ip_int >> 16) & 255}.{(ip_int >> 8) & 255}.{ip_int & 255}"


def get_network_type_for_ip(ip, seed=0):
    """
    Determines the network type for a given IP address.

    Args:
        ip (str): The IP address to evaluate.
        seed (int, optional): A seed value to alter the network type selection. Defaults to 0.

    Returns:
        TipoRed: The network type corresponding to the given IP address.
    """
    ip_int = ip_to_int(ip)
    num = ((ip_int ^ seed) & 0x7FFFFFFF) % len(TipoRed)
    return TipoRed[num]

def is_reserved(ip):
    """
    Check if the given IP address is within a reserved IP range.

    Args:
        ip (str): The IP address to check, in dotted-decimal notation (e.g., '192.168.1.1').

    Returns:
        bool: True if the IP address is within any reserved IP range, False otherwise.
    """
    ip_int = ip_to_int(ip)
    for key in reserved_ips:
        ip1 = ip_to_int(key)
        ip2 = ip_to_int(reserved_ips[key])
        if ip1 <= ip_int <= ip2:
            return True
    return False

def get_ip_for_type(net_type: int, seed=0, num_ips=1):
    """
    Generates a list of random IP addresses for a given network type.
    Args:
        net_type (int): The network type identifier, used to select the IP range.
        seed (int, optional): A seed value for deterministic IP generation. Defaults to 0.
        num_ips (int, optional): The number of IP addresses to generate. Defaults to 1.
    Returns:
        # list: A list of generated IP address strings that are not reserved.
    """

    max_num = 100

    if num_ips > max_num:
        max_num = num_ips

    ips = []

    for _ in range(max_num):
        num3 = random.randint(0, (2**31 - 1) // len(TipoRed))
        ip_text = int_to_ip((net_type + num3 * len(TipoRed)) ^ seed)
        if not is_reserved(ip_text):
            ips.append(ip_text)
        if len(ips) >= num_ips:
            break
    return ips

print(get_network_type_for_ip("85.90.248.28", 310570129))
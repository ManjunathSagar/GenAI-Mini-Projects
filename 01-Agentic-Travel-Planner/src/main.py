from crew import TravelAgentCrew


if __name__=='__main__':
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'origin': 'Bangalore',
        'destination': 'Dubai',
        'age': 35,
        'hotel_location': 'Emirates Tower',
        'flight_information': 'EK 569, leaving at July 27',
        'trip_duration': '30 days'
    }
    result = TravelAgentCrew().crew().kickoff(inputs=inputs)
    print(result)
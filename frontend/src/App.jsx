import React, { useState } from "react";
import styled from "@emotion/styled";
import "./App.css";
import axios from "axios";

const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 2rem;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const Card = styled.div`
  background: white;
  border-radius: 20px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  padding: 2rem;
  width: 100%;
  max-width: 600px;
`;

const Title = styled.h1`
  color: #2563eb;
  text-align: center;
  font-size: 2rem;
  margin-bottom: 2rem;
  font-weight: bold;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
`;

const Select = styled.select`
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
  }
`;

const TimeInput = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
  }
`;

const Button = styled.button`
  background: ${(props) => (props.disabled ? "#93c5fd" : "#2563eb")};
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  cursor: ${(props) => (props.disabled ? "not-allowed" : "pointer")};
  transition: all 0.3s ease;

  &:hover {
    background: ${(props) => (props.disabled ? "#93c5fd" : "#1d4ed8")};
  }
`;

const ResponseBox = styled.div`
  background: #f8fafc;
  border-radius: 8px;
  padding: 1.5rem;
  margin-top: 1.5rem;
  border: 1px solid #e2e8f0;
`;

function App() {
  const [startCity, setStartCity] = useState("");
  const [endCity, setEndCity] = useState("");
  const [startTime, setStartTime] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [trafficLevel, setTrafficLevel] = useState(null);
  const [trafficWait, setTrafficWait] = useState(null);
  const [passengerCount, setPassengerCount] = useState(null);

  const cities = [
    "Prishtina",
    "Prizren",
    "Peja",
    "Mitrovica",
    "Gjakova",
    "Gjilan",
    "Ferizaj",
    "Podujeva",
    "Vushtrri",
    "Deçan",
    "Rahovec",
    "Drenas",
    "Lipjan",
    "Malisheva",
    "Kaçanik",
  ];
  const isFormValid = startCity && endCity && startTime;

  const inputData = {
    start_city: 1,
    end_city: 2,
    day_of_week: 4,
    hour: 10,
    minute: 30,
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    // Simulating AI response - replace with actual API call later
    setTimeout(() => {
      (async () => {
        try {
          const str = startTime.split(":");
          const intArray = str.map(Number);

          const inputData = {
            start_city: startCity,
            end_city: endCity,
            day_of_week: "Friday",
            hour: intArray[0],
            minute: intArray[1],
          };
          const data = {
            start_city: startCity,
            end_city: endCity,
            day_of_week: "Friday",
          };

          const res1 = await axios.post(
            "http://127.0.0.1:5000/predict_traffic_level",
            inputData
          );

          const data1 = res1.data;
          if (data1.trafficLevel == 0) {
            setTrafficLevel("Light Traffic");
          } else if (data1.trafficLevel == 1) {
            setTrafficLevel("Medium Traffic");
          } else {
            setTrafficLevel("Heavy Traffic");
          }

          const res2 = await axios.post(
            "http://127.0.0.1:5000/predict_traffic_wait",
            {
              ...inputData,
              traffic_level: data1.traffic_level,
            }
          );
          const data2 = res2.data;
          setTrafficWait(data2.traffic_wait_minutes);

          const res3 = await axios.post(
            "http://127.0.0.1:5000/predict_passenger_count",
            data
          );
          const data3 = res3.data;
          setPassengerCount(data3.passenger_count);

          setResponse(
            `Route analysis from ${startCity} to ${endCity} starting at ${startTime}`
          );
        } catch (error) {
          console.error("Error during prediction:", error);
        } finally {
          setIsLoading(false);
        }
      })();
    }, 1500);
  };

  async function fetchTrafficData() {
    try {
      // 1. Predict traffic_level
      const res1 = await axios.post(
        "http://127.0.0.1:5000/predict_traffic_level",
        inputData
      );
      const data1 = res1.data;
      setTrafficLevel(data1.traffic_level);

      // 2. Predict traffic_wait_minutes
      const res2 = await fetch("/predict_traffic_wait", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...inputData,
          traffic_level: data1.traffic_level,
        }),
      });
      const data2 = await res2.json();
      setTrafficWait(data2.traffic_wait_minutes);

      // 3. Predict passenger_count
      const res3 = await fetch("/predict_passenger_count", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...inputData,
          traffic_level: data1.traffic_level,
          traffic_wait_minutes: data2.traffic_wait_minutes,
        }),
      });
      const data3 = await res3.json();
      setPassengerCount(data3.passenger_count);
    } catch (error) {
      console.error("Error fetching traffic data:", error);
    }
  }

  return (
    <Container>
      <Card>
        <Title>AI Traffic Management Assistant</Title>
        <Form onSubmit={handleSubmit}>
          <InputGroup>
            <Label>Starting City</Label>
            <Select
              value={startCity}
              onChange={(e) => setStartCity(e.target.value)}
              required
            >
              <option value="">Select a city</option>
              {cities.map((city) => (
                <option key={city} value={city}>
                  {city}
                </option>
              ))}
            </Select>
          </InputGroup>

          <InputGroup>
            <Label>Destination City</Label>
            <Select
              value={endCity}
              onChange={(e) => setEndCity(e.target.value)}
              required
            >
              <option value="">Select a city</option>
              {cities.map((city) => (
                <option key={city} value={city}>
                  {city}
                </option>
              ))}
            </Select>
          </InputGroup>

          <InputGroup>
            <Label>Starting Time</Label>
            <TimeInput
              type="time"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              required
            />
          </InputGroup>

          <Button type="submit" disabled={isLoading || !isFormValid}>
            {isLoading ? "Processing..." : "Get Route Analysis"}
          </Button>
        </Form>

        {response && (
          <ResponseBox>
            <h2 className="text-lg font-semibold mb-2">Analysis:</h2>
            <p className="text-gray-700 leading-relaxed">{response}</p>

            {trafficLevel !== null && <p>Traffic Level: {trafficLevel}</p>}
            {trafficWait !== null && (
              <p>Traffic Wait Time: {trafficWait} minutes</p>
            )}
            {passengerCount !== null && (
              <p>Passenger Count: {passengerCount}</p>
            )}
          </ResponseBox>
        )}
      </Card>
    </Container>
  );

  // return (
  //   <div>
  //     <button onClick={fetchTrafficData}>Get Traffic Info</button>
  //     {trafficLevel !== null && <p>Traffic Level: {trafficLevel}</p>}
  //     {trafficWait !== null && <p>Traffic Wait Time: {trafficWait} minutes</p>}
  //     {passengerCount !== null && <p>Passenger Count: {passengerCount}</p>}
  //   </div>
  // );
}

export default App;

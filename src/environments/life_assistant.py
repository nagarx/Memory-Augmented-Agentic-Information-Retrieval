"""
Life Assistant environment for Agentic IR.

This module implements an environment for a life assistant application.
"""

import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import datetime

from src.core.agent import Environment
from src.core.policy import Action
from src.core.state import InformationState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifeAssistantEnvironment(Environment):
    """
    Environment for a life assistant application.
    
    This environment simulates user requests and interactions with a life assistant.
    """
    
    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the life assistant environment.
        
        Args:
            user_profile: Optional user profile with preferences and settings
        """
        super().__init__()
        self.user_profile = user_profile or self._default_user_profile()
        self.current_state: Optional[InformationState] = None
        self.available_actions = [
            "search_info",
            "check_weather",
            "check_calendar",
            "set_reminder",
            "recommend_restaurant",
            "plan_trip",
            "send_message",
            "ask_clarification"
        ]
        
        # Mock data for the environment
        self.calendar: List[Dict[str, Any]] = self._get_mock_calendar()
        self.weather: Dict[str, Any] = self._get_mock_weather()
        self.restaurants: List[Dict[str, Any]] = self._get_mock_restaurants()
        
        logger.info("Initialized LifeAssistantEnvironment")
    
    def reset(self) -> InformationState:
        """
        Reset the environment to an initial state.
        
        Returns:
            The initial state
        """
        # Create a starting state with a user query
        query = self._get_random_query()
        
        state = InformationState(
            id=str(uuid.uuid4()),
            text=query,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {},
                "query_type": self._get_query_type(query)
            }
        )
        
        self.current_state = state
        logger.info(f"Reset environment with query: {query}")
        
        return state
    
    def step(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by applying an action.
        
        Args:
            action: The action to apply
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment not reset")
        
        logger.info(f"Taking action: {action.name}")
        
        # Handle the action and generate the next state
        if action.name == "search_info":
            next_state, reward, done, info = self._handle_search_info(action)
        elif action.name == "check_weather":
            next_state, reward, done, info = self._handle_check_weather(action)
        elif action.name == "check_calendar":
            next_state, reward, done, info = self._handle_check_calendar(action)
        elif action.name == "set_reminder":
            next_state, reward, done, info = self._handle_set_reminder(action)
        elif action.name == "recommend_restaurant":
            next_state, reward, done, info = self._handle_recommend_restaurant(action)
        elif action.name == "plan_trip":
            next_state, reward, done, info = self._handle_plan_trip(action)
        elif action.name == "send_message":
            next_state, reward, done, info = self._handle_send_message(action)
        elif action.name == "ask_clarification":
            next_state, reward, done, info = self._handle_ask_clarification(action)
        else:
            # Unknown action
            next_state = InformationState(
                id=str(uuid.uuid4()),
                text=f"I don't know how to {action.name}. Can you please try a different action?",
                parent_id=self.current_state.id,
                available_actions=self.available_actions,
                timestamp=time.time(),
                data={
                    "user_profile": self.user_profile,
                    "context": self.current_state.data.get("context", {}),
                    "query_type": "unknown"
                }
            )
            reward = -0.1  # Penalty for unknown action
            done = False
            info = {"error": f"Unknown action: {action.name}"}
        
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def render(self) -> str:
        """
        Render the environment for visualization.
        
        Returns:
            A string representation of the environment
        """
        if self.current_state is None:
            return "Environment not reset"
            
        return f"Current state: {self.current_state.text}"
    
    def _default_user_profile(self) -> Dict[str, Any]:
        """
        Create a default user profile.
        
        Returns:
            A dictionary with user profile information
        """
        return {
            "name": "Jane",
            "location": "San Francisco, CA",
            "preferences": {
                "food": ["Italian", "Japanese", "Mexican"],
                "activities": ["Hiking", "Reading", "Movies"],
                "transportation": ["Car", "Public Transit"]
            },
            "work_hours": {
                "start": "09:00",
                "end": "17:00"
            }
        }
    
    def _get_mock_calendar(self) -> List[Dict[str, Any]]:
        """
        Get mock calendar data.
        
        Returns:
            A list of calendar events
        """
        today = datetime.datetime.now()
        tomorrow = today + datetime.timedelta(days=1)
        
        return [
            {
                "title": "Team Meeting",
                "start": today.replace(hour=10, minute=0).isoformat(),
                "end": today.replace(hour=11, minute=0).isoformat(),
                "location": "Conference Room A"
            },
            {
                "title": "Lunch with Alex",
                "start": today.replace(hour=12, minute=30).isoformat(),
                "end": today.replace(hour=13, minute=30).isoformat(),
                "location": "Cafe Deluxe"
            },
            {
                "title": "Doctor Appointment",
                "start": tomorrow.replace(hour=14, minute=0).isoformat(),
                "end": tomorrow.replace(hour=15, minute=0).isoformat(),
                "location": "City Medical Center"
            }
        ]
    
    def _get_mock_weather(self) -> Dict[str, Any]:
        """
        Get mock weather data.
        
        Returns:
            Weather information
        """
        return {
            "current": {
                "temperature": 72,
                "description": "Sunny",
                "humidity": 45,
                "wind_speed": 5
            },
            "forecast": [
                {
                    "day": "Today",
                    "high": 75,
                    "low": 60,
                    "description": "Sunny"
                },
                {
                    "day": "Tomorrow",
                    "high": 70,
                    "low": 58,
                    "description": "Partly Cloudy"
                },
                {
                    "day": "Day After",
                    "high": 65,
                    "low": 55,
                    "description": "Rain"
                }
            ]
        }
    
    def _get_mock_restaurants(self) -> List[Dict[str, Any]]:
        """
        Get mock restaurant data.
        
        Returns:
            A list of restaurants
        """
        return [
            {
                "name": "Bella Italia",
                "cuisine": "Italian",
                "rating": 4.5,
                "price": "$$",
                "location": "123 Main St, San Francisco, CA"
            },
            {
                "name": "Sakura Sushi",
                "cuisine": "Japanese",
                "rating": 4.7,
                "price": "$$$",
                "location": "456 Oak St, San Francisco, CA"
            },
            {
                "name": "El Camino",
                "cuisine": "Mexican",
                "rating": 4.2,
                "price": "$$",
                "location": "789 Pine St, San Francisco, CA"
            },
            {
                "name": "Golden Dragon",
                "cuisine": "Chinese",
                "rating": 4.3,
                "price": "$$",
                "location": "321 Cedar St, San Francisco, CA"
            },
            {
                "name": "Parisian Bistro",
                "cuisine": "French",
                "rating": 4.6,
                "price": "$$$",
                "location": "654 Elm St, San Francisco, CA"
            }
        ]
    
    def _get_random_query(self) -> str:
        """
        Get a random user query.
        
        Returns:
            A user query string
        """
        queries = [
            "What's the weather like today?",
            "Do I have any meetings today?",
            "Can you recommend a good restaurant for dinner?",
            "Set a reminder for my doctor's appointment tomorrow",
            "I need to plan a trip to New York next week",
            "Send a message to Alex about lunch",
            "What's the traffic like for my commute?",
            "Can you find information about hiking trails nearby?"
        ]
        
        import random
        return random.choice(queries)
    
    def _get_query_type(self, query: str) -> str:
        """
        Determine the type of query.
        
        Args:
            query: The user query
            
        Returns:
            The query type
        """
        query_lower = query.lower()
        
        if "weather" in query_lower:
            return "weather"
        elif "meeting" in query_lower or "calendar" in query_lower or "schedule" in query_lower:
            return "calendar"
        elif "restaurant" in query_lower or "dinner" in query_lower or "lunch" in query_lower:
            return "restaurant"
        elif "reminder" in query_lower or "appointment" in query_lower:
            return "reminder"
        elif "trip" in query_lower or "travel" in query_lower:
            return "trip"
        elif "message" in query_lower or "send" in query_lower or "text" in query_lower:
            return "message"
        elif "traffic" in query_lower or "commute" in query_lower:
            return "traffic"
        elif "find" in query_lower or "information" in query_lower or "search" in query_lower:
            return "search"
        else:
            return "other"
    
    def _handle_search_info(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the search_info action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        query = action.parameters.get("query", self.current_state.text)
        
        response = f"I found the following information about '{query}':\n\n"
        response += "1. According to sources, {query} is a topic of interest to many people.\n"
        response += "2. There are several resources available that provide more details.\n"
        response += "3. You might want to check out some related information as well."
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {**self.current_state.data.get("context", {}), "last_search": query},
                "query_type": "search_response"
            }
        )
        
        # Determine if the task is complete
        done = True
        reward = 0.5  # Moderate reward for providing information
        info = {"action": "search_info", "query": query}
        
        return next_state, reward, done, info
    
    def _handle_check_weather(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the check_weather action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        location = action.parameters.get("location", self.user_profile["location"])
        
        weather = self.weather
        current = weather["current"]
        forecast = weather["forecast"]
        
        response = f"Weather for {location}:\n\n"
        response += f"Current conditions: {current['temperature']}°F, {current['description']}\n"
        response += f"Humidity: {current['humidity']}%, Wind: {current['wind_speed']} mph\n\n"
        response += "Forecast:\n"
        
        for day in forecast:
            response += f"- {day['day']}: {day['description']}, High {day['high']}°F, Low {day['low']}°F\n"
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {**self.current_state.data.get("context", {}), "weather": weather},
                "query_type": "weather_response"
            }
        )
        
        # Weather queries are typically complete after providing the information
        done = True
        reward = 0.7  # Good reward for providing weather information
        info = {"action": "check_weather", "location": location}
        
        return next_state, reward, done, info
    
    def _handle_check_calendar(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the check_calendar action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        date_str = action.parameters.get("date", "today")
        
        if date_str == "today":
            date = datetime.datetime.now()
        elif date_str == "tomorrow":
            date = datetime.datetime.now() + datetime.timedelta(days=1)
        else:
            try:
                date = datetime.datetime.fromisoformat(date_str)
            except ValueError:
                date = datetime.datetime.now()
        
        # Filter events for the requested date
        events = []
        for event in self.calendar:
            event_start = datetime.datetime.fromisoformat(event["start"].split('T')[0])
            if event_start.date() == date.date():
                events.append(event)
        
        if events:
            response = f"Calendar for {date.strftime('%A, %B %d')}:\n\n"
            for event in events:
                start_time = datetime.datetime.fromisoformat(event["start"]).strftime("%I:%M %p")
                end_time = datetime.datetime.fromisoformat(event["end"]).strftime("%I:%M %p")
                response += f"- {start_time} - {end_time}: {event['title']} at {event['location']}\n"
        else:
            response = f"You have no events scheduled for {date.strftime('%A, %B %d')}."
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {**self.current_state.data.get("context", {}), "calendar_events": events},
                "query_type": "calendar_response"
            }
        )
        
        # Calendar queries are typically complete after providing the information
        done = True
        reward = 0.7  # Good reward for providing calendar information
        info = {"action": "check_calendar", "date": date_str}
        
        return next_state, reward, done, info
    
    def _handle_set_reminder(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the set_reminder action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        text = action.parameters.get("text", "")
        time_str = action.parameters.get("time", "")
        date_str = action.parameters.get("date", "today")
        
        if not text:
            response = "I need to know what to remind you about. Can you please provide the reminder text?"
            done = False
            reward = 0.0  # Neutral reward for asking clarification
        elif not time_str:
            response = f"I'll set a reminder for '{text}'. What time would you like to be reminded?"
            done = False
            reward = 0.1  # Small reward for partial progress
        else:
            response = f"I've set a reminder for '{text}' on {date_str} at {time_str}."
            done = True
            reward = 0.8  # Good reward for setting a reminder
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {
                    **self.current_state.data.get("context", {}),
                    "reminder": {"text": text, "time": time_str, "date": date_str}
                },
                "query_type": "reminder_response"
            }
        )
        
        info = {
            "action": "set_reminder",
            "text": text,
            "time": time_str,
            "date": date_str
        }
        
        return next_state, reward, done, info
    
    def _handle_recommend_restaurant(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the recommend_restaurant action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        cuisine = action.parameters.get("cuisine", "")
        price = action.parameters.get("price", "")
        
        # Filter restaurants based on criteria
        filtered = self.restaurants
        
        if cuisine:
            filtered = [r for r in filtered if r["cuisine"].lower() == cuisine.lower()]
        
        if price:
            filtered = [r for r in filtered if r["price"] == price]
        
        # If no restaurants match the criteria, return all restaurants
        if not filtered:
            filtered = self.restaurants
            
        # Sort by rating
        filtered.sort(key=lambda r: r["rating"], reverse=True)
        
        if filtered:
            response = "Here are some restaurant recommendations for you:\n\n"
            for i, restaurant in enumerate(filtered[:3], 1):
                response += f"{i}. {restaurant['name']} - {restaurant['cuisine']} cuisine\n"
                response += f"   Rating: {restaurant['rating']}/5, Price: {restaurant['price']}\n"
                response += f"   Location: {restaurant['location']}\n\n"
        else:
            response = "I couldn't find any restaurants matching your criteria."
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {
                    **self.current_state.data.get("context", {}),
                    "restaurants": filtered[:3]
                },
                "query_type": "restaurant_response"
            }
        )
        
        # Restaurant recommendations are typically complete after providing options
        done = True
        reward = 0.7  # Good reward for providing recommendations
        info = {
            "action": "recommend_restaurant",
            "cuisine": cuisine,
            "price": price
        }
        
        return next_state, reward, done, info
    
    def _handle_plan_trip(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the plan_trip action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        destination = action.parameters.get("destination", "")
        start_date = action.parameters.get("start_date", "")
        end_date = action.parameters.get("end_date", "")
        
        if not destination:
            response = "I'd be happy to help you plan a trip. Where would you like to go?"
            done = False
            reward = 0.0  # Neutral reward for asking clarification
        elif not start_date:
            response = f"Planning a trip to {destination}. When would you like to depart?"
            done = False
            reward = 0.1  # Small reward for partial progress
        elif not end_date:
            response = f"Planning a trip to {destination} starting on {start_date}. When would you like to return?"
            done = False
            reward = 0.2  # Small reward for more progress
        else:
            response = f"Here's a draft itinerary for your trip to {destination} from {start_date} to {end_date}:\n\n"
            response += "Day 1: Arrival and check-in\n"
            response += "- Flight arrival (time to be determined)\n"
            response += "- Transfer to accommodation\n"
            response += "- Evening: Explore nearby attractions\n\n"
            response += "Day 2: City Exploration\n"
            response += "- Morning: Visit key landmarks\n"
            response += "- Afternoon: Museum or cultural experience\n"
            response += "- Evening: Local cuisine dinner\n\n"
            response += "Day 3: Adventure Day\n"
            response += "- Full day excursion to natural attractions\n"
            response += "- Evening: Entertainment or relaxation\n\n"
            response += "Final Day: Departure\n"
            response += "- Morning: Last-minute shopping or sightseeing\n"
            response += "- Check-out and transfer to airport\n"
            response += "- Return flight (time to be determined)\n\n"
            response += "Would you like me to add more details or make any changes to this itinerary?"
            
            done = True
            reward = 0.9  # Excellent reward for a complete trip plan
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {
                    **self.current_state.data.get("context", {}),
                    "trip": {"destination": destination, "start_date": start_date, "end_date": end_date}
                },
                "query_type": "trip_response"
            }
        )
        
        info = {
            "action": "plan_trip",
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return next_state, reward, done, info
    
    def _handle_send_message(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the send_message action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        recipient = action.parameters.get("recipient", "")
        message = action.parameters.get("message", "")
        
        if not recipient:
            response = "Who would you like to send a message to?"
            done = False
            reward = 0.0  # Neutral reward for asking clarification
        elif not message:
            response = f"What message would you like to send to {recipient}?"
            done = False
            reward = 0.1  # Small reward for partial progress
        else:
            response = f"I've sent your message to {recipient}: \"{message}\""
            done = True
            reward = 0.6  # Good reward for sending a message
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=response,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": {
                    **self.current_state.data.get("context", {}),
                    "message": {"recipient": recipient, "text": message}
                },
                "query_type": "message_response"
            }
        )
        
        info = {
            "action": "send_message",
            "recipient": recipient,
            "message": message
        }
        
        return next_state, reward, done, info
    
    def _handle_ask_clarification(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Handle the ask_clarification action.
        
        Args:
            action: The action with parameters
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        question = action.parameters.get("question", "Can you please clarify what you're looking for?")
        
        next_state = InformationState(
            id=str(uuid.uuid4()),
            text=question,
            parent_id=self.current_state.id,
            available_actions=self.available_actions,
            timestamp=time.time(),
            data={
                "user_profile": self.user_profile,
                "context": self.current_state.data.get("context", {}),
                "query_type": "clarification_request"
            }
        )
        
        done = False
        reward = 0.1  # Small reward for asking for clarification when needed
        info = {"action": "ask_clarification", "question": question}
        
        return next_state, reward, done, info 
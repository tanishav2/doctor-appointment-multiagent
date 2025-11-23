import pandas as pd
from typing import Literal
from langchain_core.tools import tool


# -----------------------------------------
# 1. CHECK AVAILABILITY BY DOCTOR
# -----------------------------------------
@tool
def check_availability_by_doctor(
    date: str,
    doctor_name: Literal[
        'kevin anderson','robert martinez','susan davis','daniel miller',
        'sarah wilson','michael green','lisa brown','jane smith',
        'emily johnson','john doe'
    ]
):
    """
    Check availability for a specific doctor on a given date.
    date: "DD-MM-YYYY"
    """
    df = pd.read_csv("data/doctor_availability.csv")
    df['date_slot_time'] = df['date_slot'].apply(lambda x: x.split(" ")[-1])

    rows = list(
        df[
            (df['date_slot'].apply(lambda x: x.split(" ")[0]) == date)
            & (df['doctor_name'] == doctor_name)
            & (df['is_available'] == True)
        ]['date_slot_time']
    )

    if not rows:
        return "No availability in the entire day."

    return (
        f"Availability for {date}\nAvailable slots: "
        + ", ".join(rows)
    )


# -----------------------------------------
# 2. CHECK SPECIFIC TIME SLOT
# -----------------------------------------
@tool
def check_specific_slot(
    date: str,  # "DD-MM-YYYY"
    time: str,  # "HH:MM" format (24-hour)
    doctor_name: str = None,  # Optional: if provided, check for specific doctor
    specialization: str = None  # Optional: if provided, check for specialization
):
    """
    Check if a specific time slot is available and return nearby alternatives if not.
    date: "DD-MM-YYYY", time: "HH:MM" (24-hour format like "20:00" for 8PM)
    Returns: Status of the slot and nearby alternatives.
    """
    df = pd.read_csv("data/doctor_availability.csv")
    
    from datetime import datetime, timedelta
    
    # Convert time to match CSV format (HH:MM)
    try:
        time_obj = datetime.strptime(time, "%H:%M")
        time_str = time_obj.strftime("%H:%M")
    except:
        # Try to parse as 12-hour format
        try:
            time_obj = datetime.strptime(time, "%I:%M %p")
            time_str = time_obj.strftime("%H:%M")
        except:
            return f"Invalid time format. Please use HH:MM (24-hour) or HH:MM AM/PM format."
    
    # Format the full datetime slot - CSV uses "DD-MM-YYYY HH:MM" format
    date_slot_str = f"{date} {time_str}"
    
    # The CSV format can be "DD-MM-YYYY HH:MM" (with colon) or "DD-MM-YYYY H.MM" (with dot)
    # Try both formats when checking
    formatted_slot_colon = date_slot_str
    formatted_slot_dot = date_slot_str.replace(":", ".")
    
    def convert_to_am_pm(time_str):
        """Convert 24-hour time to natural 12-hour AM/PM format (simple and casual)"""
        try:
            # Handle both colon and dot formats
            time_clean = str(time_str).replace(".", ":")
            hours, minutes = map(int, time_clean.split(":"))
            period = "AM" if hours < 12 else "PM"
            hours = hours % 12 or 12
            # If minutes are 00, just show hour (e.g., "8 AM" not "8:00 AM")
            # If minutes are 30, show "8:30 AM"
            if minutes == 0:
                return f"{hours} {period}"
            else:
                return f"{hours}:{minutes:02d} {period}"
        except:
            return time_str
    
    # Check if specific slot is available
    if doctor_name:
        # Check for specific doctor - try both formats
        slot = df[
            ((df['date_slot'] == formatted_slot_colon) | (df['date_slot'] == formatted_slot_dot))
            & (df['doctor_name'] == doctor_name.lower())
            & (df['is_available'] == True)
        ]
        
        if len(slot) > 0:
            time_display = convert_to_am_pm(time_str)
            return f"YES_AVAILABLE: That time slot is available for Dr. {doctor_name.title()} on {date} at {time_display}."
        
        # Get nearby alternatives for same doctor
        same_date_slots = df[
            (df['date_slot'].apply(lambda x: x.split(" ")[0]) == date)
            & (df['doctor_name'] == doctor_name.lower())
            & (df['is_available'] == True)
        ]
        
        if len(same_date_slots) > 0:
            alternatives = same_date_slots['date_slot'].apply(lambda x: x.split(" ")[-1]).tolist()[:5]
            # Handle both colon and dot formats, convert to natural format
            alt_times = [convert_to_am_pm(t.replace(".", ":")) if "." in t else convert_to_am_pm(t) for t in alternatives]
            return f"NOT_AVAILABLE: The requested time is not available. Here are other times on {date} for Dr. {doctor_name.title()}: {', '.join(alt_times)}"
        else:
            return f"NOT_AVAILABLE: No slots available for Dr. {doctor_name.title()} on {date}."
    
    elif specialization:
        # Check for specialization - try both formats
        slot = df[
            ((df['date_slot'] == formatted_slot_colon) | (df['date_slot'] == formatted_slot_dot))
            & (df['specialization'] == specialization)
            & (df['is_available'] == True)
        ]
        
        if len(slot) > 0:
            doctors = slot['doctor_name'].unique().tolist()
            time_display = convert_to_am_pm(time_str)
            doctor_list = ', '.join([f"Dr. {d.title()}" for d in doctors])
            return f"YES_AVAILABLE: That time is available on {date} at {time_display} with {doctor_list}."
        
        # Get nearby alternatives and doctors
        same_date_slots = df[
            (df['date_slot'].apply(lambda x: x.split(" ")[0]) == date)
            & (df['specialization'] == specialization)
            & (df['is_available'] == True)
        ]
        
        if len(same_date_slots) > 0:
            # Group by doctor and get top slots
            grouped = same_date_slots.groupby('doctor_name')['date_slot'].apply(
                lambda x: [t.split(" ")[-1] for t in x[:3]]
            ).to_dict()
            
            output = f"NOT_AVAILABLE: That time isn't available. Here are other options on {date}: "
            doctor_times = []
            for doc, times in list(grouped.items())[:3]:
                time_list = [convert_to_am_pm(t.replace(".", ":")) if "." in t else convert_to_am_pm(t) for t in times]
                doctor_times.append(f"Dr. {doc.title()} - {', '.join(time_list)}")
            output += " | ".join(doctor_times)
            return output
        else:
            return f"NOT_AVAILABLE: No slots available for {specialization} on {date}."
    
    else:
        return "Please specify either doctor_name or specialization to check availability."


# -----------------------------------------
# 3. CHECK AVAILABILITY BY SPECIALIZATION
# -----------------------------------------
@tool
def check_availability_by_specialization(
    date: str,
    specialization: Literal[
        "general_dentist", "cosmetic_dentist", "prosthodontist", 
        "pediatric_dentist", "emergency_dentist", 
        "oral_surgeon", "orthodontist"
    ]
):
    """
    Check availability for a specialization on a given date.
    date: "DD-MM-YYYY"
    """

    df = pd.read_csv("data/doctor_availability.csv")
    df['date_slot_time'] = df['date_slot'].apply(lambda x: x.split(' ')[-1])

    rows = df[
        (df['date_slot'].apply(lambda x: x.split(" ")[0]) == date)
        & (df['specialization'] == specialization)
        & (df['is_available'] == True)
    ].groupby(['specialization', 'doctor_name'])['date_slot_time'].apply(list).reset_index(name='available_slots')

    if len(rows) == 0:
        return "No availability in the entire day."

    def convert_to_am_pm(time_str):
        """Convert 24-hour time to natural 12-hour AM/PM format"""
        try:
            time_clean = str(time_str).replace(".", ":")
            hours, minutes = map(int, time_clean.split(":"))
            period = "AM" if hours < 12 else "PM"
            hours = hours % 12 or 12
            return f"{hours}:{minutes:02d} {period}" if minutes > 0 else f"{hours} {period}"
        except:
            return time_str

    output = f"Availability for {date}\n"
    for row in rows.values:
        doctor_name = row[1].title()
        time_slots = [convert_to_am_pm(t.replace(".", ":")) if "." in str(t) else convert_to_am_pm(t) for t in row[2][:5]]  # Limit to 5 slots
        output += f"Dr. {doctor_name}: {', '.join(time_slots)}\n"

    return output


# -----------------------------------------
# 4. SET APPOINTMENT
# -----------------------------------------
@tool
def set_appointment(
    date: str,      # "DD-MM-YYYY HH:MM"
    id_number: str, # patient ID
    doctor_name: Literal[
        'kevin anderson','robert martinez','susan davis','daniel miller',
        'sarah wilson','michael green','lisa brown','jane smith',
        'emily johnson','john doe'
    ]
):
    """
    Set an appointment for a specific datetime.
    date must be "DD-MM-YYYY HH:MM"
    """

    df = pd.read_csv("data/doctor_availability.csv")

    from datetime import datetime
    def convert_datetime_format(dt_str):
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H:%M")
        # Return both formats to match what's in CSV
        return dt.strftime("%d-%m-%Y %H:%M"), dt.strftime("%d-%m-%Y %H.%M")

    formatted_colon, formatted_dot = convert_datetime_format(date)

    # Check availability using both formats (CSV might have either)
    case = df[
        ((df['date_slot'] == formatted_colon) | (df['date_slot'] == formatted_dot))
        & (df['doctor_name'] == doctor_name.lower())
        & (df['is_available'] == True)
    ]

    if len(case) == 0:
        return "No available appointments for that time."

    # Update using the actual format found in CSV
    actual_slot = case['date_slot'].iloc[0]  # Get the actual format from CSV
    
    df.loc[
        (df['date_slot'] == actual_slot)
        & (df['doctor_name'] == doctor_name.lower()),
        ['is_available', 'patient_to_attend']
    ] = [False, id_number]

    df.to_csv("data/doctor_availability.csv", index=False)
    return "Appointment successfully booked."


# -----------------------------------------
# 5. CANCEL APPOINTMENT
# -----------------------------------------
@tool
def cancel_appointment(
    date: str,      # "DD-MM-YYYY HH:MM"
    id_number: str,
    doctor_name: Literal[
        'kevin anderson','robert martinez','susan davis','daniel miller',
        'sarah wilson','michael green','lisa brown','jane smith',
        'emily johnson','john doe'
    ]
):
    """
    Cancel an existing appointment.
    """

    df = pd.read_csv("data/doctor_availability.csv")

    case = df[
        (df['date_slot'] == date)
        & (df['patient_to_attend'] == id_number)
        & (df['doctor_name'] == doctor_name)
    ]

    if len(case) == 0:
        return "You do not have any appointment with these details."

    df.loc[
        (df['date_slot'] == date)
        & (df['patient_to_attend'] == id_number)
        & (df['doctor_name'] == doctor_name),
        ['is_available', 'patient_to_attend']
    ] = [True, None]

    df.to_csv("data/doctor_availability.csv", index=False)
    return "Appointment successfully cancelled."


# -----------------------------------------
# 6. RESCHEDULE APPOINTMENT
# -----------------------------------------
@tool
def reschedule_appointment(
    old_date: str,      # "DD-MM-YYYY HH:MM"
    new_date: str,      # "DD-MM-YYYY HH:MM"
    id_number: str,
    doctor_name: Literal[
        'kevin anderson','robert martinez','susan davis','daniel miller',
        'sarah wilson','michael green','lisa brown','jane smith',
        'emily johnson','john doe'
    ]
):
    """
    Reschedule appointment from old_date to new_date.
    """

    df = pd.read_csv("data/doctor_availability.csv")

    available = df[
        (df['date_slot'] == new_date)
        & (df['is_available'] == True)
        & (df['doctor_name'] == doctor_name)
    ]

    if len(available) == 0:
        return "Desired new date has no available slots."

    cancel_appointment.invoke({"date": old_date, "id_number": id_number, "doctor_name": doctor_name})
    set_appointment.invoke({"date": new_date, "id_number": id_number, "doctor_name": doctor_name})

    return "Appointment successfully rescheduled."

from fastapi import HTTPException, status
import uuid
from schemas.patient_schema import PatientCreate, PatientGet, PatientUpdate
from db.database import supabase

TABLE_NAME = "patients"

def check_uuid(id: str):
    try:
        validated_uuid = str(uuid.UUID(id))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format")
    return validated_uuid

def get_patients_service(limit: int, offset: int):
    columns = ','.join(PatientGet.model_fields.keys())
    result = supabase.table(TABLE_NAME).select(columns).range(offset, offset + limit - 1).execute()
    return {"data": result.data, "count": len(result.data)}

def get_random_patient_service():
    rand_uuid = str(uuid.uuid4())
    columns = ','.join(PatientGet.model_fields.keys())
    result = supabase.table(TABLE_NAME).select(columns).order("id").gte("id", rand_uuid).limit(1).execute()
    if not result.data:
        result = supabase.table(TABLE_NAME).select(columns).order("id").limit(1).execute()
    return result.data[0]

def get_patient_service(patient_id: str, user: dict):
    patient_uuid = check_uuid(patient_id)
    columns = ','.join(PatientGet.model_fields.keys())

    if user["role"] == "admin":
        result = supabase.table(TABLE_NAME).select(columns).eq("id", patient_uuid).execute()
    else:
        result = supabase.table(TABLE_NAME).select(columns).eq("id", patient_uuid).eq("user_id", user["user_id"]).execute()

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return result.data[0]

def get_patients_by_user(user_id: str):
    user_uuid = check_uuid(user_id)
    columns = ','.join(PatientGet.model_fields.keys())
    result = supabase.table(TABLE_NAME).select(columns).eq("user_id", user_uuid).execute()
    return result.data

def create_patient_service(new_patient: PatientCreate, user_id: str):
    patient_data = new_patient.model_dump()
    patient_data["user_id"] = user_id

    result = supabase.table(TABLE_NAME).insert(patient_data).execute()

    if not result.data:
        raise Exception("Failed to create patient")
    return result.data[0]

def update_patient_service(patient_id: str, patient: PatientUpdate, user: dict):
    patient_uuid = check_uuid(patient_id)
    update_data = patient.model_dump(exclude_unset=True)
    columns = ','.join(PatientGet.model_fields.keys())

    if user["role"] == "admin":
        result = supabase.table(TABLE_NAME).select(columns).eq("id", patient_uuid).execute()
    else:
        result = supabase.table(TABLE_NAME).select(columns).eq("id", patient_uuid).eq("user_id", user["user_id"]).execute()

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    
    result = supabase.table(TABLE_NAME).update(update_data).eq("id", patient_uuid).execute()
    return result.data[0]

def delete_patient_service(patient_id: str, user: dict):
    patient_uuid = check_uuid(patient_id)

    if user["role"] == "admin":
        result = supabase.table(TABLE_NAME).delete().eq("id", patient_uuid).execute()
    else:
        result = supabase.table(TABLE_NAME).delete().eq("id", patient_uuid).eq("user_id", user["user_id"]).execute()
        
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return {"detail": "Delete patient successfully"}

def delete_user_patients_service(user_id: str):
    user_uuid = check_uuid(user_id)
    supabase.table(TABLE_NAME).delete().eq("user_id", user_uuid).execute()
    return {"detail": "Delete patients successfully"}

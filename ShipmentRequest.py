from pydantic import BaseModel

class ShipmentRequest(BaseModel):
    shipment_id: str
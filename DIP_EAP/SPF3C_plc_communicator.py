# -*- coding: utf-8 -*-
"""
SPF3C_plc_communicator.py - PLC communication for industrial image processing system

Contains all PLC communication logic and data conversion methods.
"""

import pymcprotocol


class PLCCommunicator:
    """PLC communication class for the industrial image processing system"""
    
    def __init__(self, config_manager):
        """Initialize PLC communicator"""
        self.config = config_manager
        self.mc = None
        self.plc_connected = False
    
    def connect_plc(self):
        """PLC connection (same logic as original file)"""
        try:
            self.mc = pymcprotocol.Type3E()
            self.mc.connect(self.config.PLC_IP_ADDRESS, self.config.PLC_PORT)
            self.plc_connected = True
            print("✅ PLC connection successful")
        except Exception as e:
            print(f"⚠️ PLC connection failed: {e}")
            print("🔄 Skipping PLC communication and running in local mode")
            self.plc_connected = False
    
    def int32_to_words(self, n: int) -> list[int]:
        """Convert 32-bit signed integer to two WORDs (low, high) (same as original file)"""
        n &= self.config.BIT32_MASK  # 32-bit two's complement
        return [n & self.config.BIT16_MASK, (n >> self.config.BIT_SHIFT_16) & self.config.BIT16_MASK]
    
    def read_plc_data(self):
        """Read PLC data with error handling"""
        try:
            d28 = self.config.PLC_DEFAULT_VALUE  # Default value
            if self.plc_connected:
                try:
                    d28 = self.mc.batchread_wordunits(headdevice=self.config.PLC_DEVICE_D28, readsize=1)[self.config.ARRAY_FIRST_INDEX]
                except Exception as e:
                    print(f"⚠️ PLC read error: {e}")
                    d28 = self.config.PLC_DEFAULT_VALUE  # Use default value on error
            
            return d28
        except Exception as e:
            print(f"⚠️ PLC processing error: {e}")
            return self.config.PLC_DEFAULT_VALUE
    
    def write_plc_data(self, send_value: int, number_of_rows: int, success: bool):
        """Write PLC data with error handling"""
        try:
            if self.plc_connected:
                words = self.int32_to_words(send_value)
                self.mc.batchwrite_wordunits(headdevice=self.config.PLC_DEVICE_D29, values=words)
                self.mc.batchwrite_wordunits(headdevice=self.config.PLC_DEVICE_D14, values=[number_of_rows])
                
                if success:
                    self.mc.batchwrite_wordunits(headdevice=self.config.PLC_DEVICE_D28, values=[self.config.PLC_SUCCESS_VALUE])
                else:
                    self.mc.batchwrite_wordunits(headdevice=self.config.PLC_DEVICE_D28, values=[self.config.PLC_ERROR_VALUE])
        except Exception as e:
            print(f"⚠️ PLC write error: {e}")
    
    def is_connected(self):
        """Check if PLC is connected"""
        return self.plc_connected
    
    def get_connection_status(self):
        """Get PLC connection status"""
        return self.plc_connected 
# Databricks notebook source
	curl -X POST \ 
	-H "Authorization: Bearer xxxxxx" \ 
	-H "Content-Type: application/json" \ 
	-d '{"inputs":[{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE"}]}' \ 
https://adb-xxxx.azuredatabricks.net/serving-endpoints/penguings_predicitons_live/invocations
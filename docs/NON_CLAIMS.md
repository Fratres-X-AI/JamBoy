# Non-claims

JamBoy is research / simulation software. These statements are out of scope:

- **Not jam-proof.** The name is a pun (GPS jam + Game Boy). The software estimates position without GPS in simulation. It does not defeat a real jammer or spoofing attack.
- **Not flight-certified.** No airworthiness, no STANAG, no FAA/military qualification.
- **Not flight-proven.** No published hardware flight log is a release gate.
- **Not a targeting or weapons system.** No ROE, no strike, no seeker.
- **Sim is synthetic.** Dummy maps and frames are generated. Clean-sim RMSE is not a field CEP.
- **Hardware path is unvalidated.** Pi / Pixhawk / IMX296 docs are a drop-in checklist, not a completed integration.
- **Windows may not run the full gate.** `rasterio` / GDAL can fail under WDAC. Linux CI is the public proof. `scripts/check_laptop.ps1` runs a non-geo subset when maps cannot load.

Use in accordance with applicable law and export controls.

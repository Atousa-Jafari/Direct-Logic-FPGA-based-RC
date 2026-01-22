# ESN (Vivado 2023.2 and Vitis 2023.2)
1. Generate bitstream in the vivado project
2. Export hardware with bitstream included
3. In Vitis IDE, generate the platform using the above hardware file. 
4. Create an application template with the hello world example. 
5. Replace helloworld.c file with the ESN.c source file. 
6. Include the data.h file in the sources. 
7. Build the platform and build the application with the platform to generate the fsbl and elf. 
8. Program the board. 
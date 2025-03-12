To run the code, first compile the project:

`make clean`
`make`

Then, run with the following format:
./main <input_filename> [output_filename] [--optimizations=optimization1,optimization2,...] [--list-all-solutions]

where <input_filename> is the file containing the graph, [output_filename] is the (optional) output file (the result will be printed to stdout by default), and [optimizations] are one of default (not altering the vertices order from the graph file, highest-order (prioritize higher degree vertices first) or lowest-order (prioritize lower degree first) and/or bridges (not supported yet). [--list-all-solutions] flag is also optional, which when enabled will print all solutions instead of finding just one solution.

Example usage:

./main graph.dat res.txt --optimizations=highest-order --list-all-solutions (valid)
./main graph.dat res.txt --optimizations=default,highest-order (invalid since only one among default, highest-order and lowest-order can be selected)
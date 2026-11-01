# Runs every method of ml_representations for one epoch on a tiny dataset and
# checks that a result JSON is written. Usage: cmake -DEXE=... -DOUT=... -P this
file(MAKE_DIRECTORY "${OUT}")
foreach(method uniform adaptive gaussian adaptive_gaussian)
    execute_process(
        COMMAND "${EXE}" --method ${method} --seed 1 --epochs 1
                --train-per-class 3 --test-per-class 2 --points 512
                --output "${OUT}/${method}.json"
        RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "ml_representations --method ${method} failed: ${rc}")
    endif()
    if(NOT EXISTS "${OUT}/${method}.json")
        message(FATAL_ERROR "ml_representations --method ${method} wrote no result")
    endif()
endforeach()

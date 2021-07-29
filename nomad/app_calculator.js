const calculator = {
    plus: function(a, b) {
        return a+b
    },
    minus: function(a, b) {
        return a-b
    },
    times: function(a, b) {
        return a * b
    },
    divide: function(a, b) {
        return a / b
    },
    power: function(a, b) {
        a ** b
    }
}

const plusResult = calculator.plus(3, 5)
const minusResult = calculator.minus(3, plusResult)
const timesResult = calcualtor.times(7, minusResult)
const divideResult = calculator.divide(12, timesResult)
const powerResult = calculator.power(divideResult, 3)
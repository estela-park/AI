const age = parseInt(prompt('How old are you?'))

if (isNaN(age) || age < 0) {
    console.log('Please write a positive number')
} else if (age < 18) {
    console.log('You are too young')
} else if ( age >= 18 && age <= 50) {
    console.log('You can drink')
} else if (age === 100) {
    console.log("I bet you're wise")
} else if (age !== 99) {
    console.log('You are almost there.')
}

if (condition) {
    /// codes to execute for condition === true
} else if (another_condition) {
    /// codes to execute for condition === false && another condition === true
} else {
    /// codes to execute for condition === false && another condition === false
}
/// and &&, or ||
/// else and else if are optional
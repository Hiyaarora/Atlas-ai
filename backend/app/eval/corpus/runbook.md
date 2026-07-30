# Platform Operations Runbook

## Severity definitions

An incident is classified SEV1 when customer-facing writes are failing, SEV2
when a subsystem is degraded but the product remains usable, and SEV3 for
issues with a workaround that can wait for business hours. The classification
is set by the responder and may be revised upward at any time, but lowering it
requires agreement from the incident commander.

## Paging and escalation

The primary on-call engineer is paged immediately. If the page is not
acknowledged within fifteen minutes, the alert escalates to the secondary. A
further ten minutes without acknowledgement escalates to the platform lead,
who may convene a wider response. Rotations hand over on Monday mornings and
the outgoing engineer is expected to brief the incoming one on anything still
open.

## Roles during an incident

The incident commander coordinates and does not debug. The moment the person
directing the response is also the person reading logs, updates stop and
parallel work collides. A separate communications lead owns the status page
and customer-facing updates, and a scribe records the timeline as it happens
rather than reconstructing it afterwards from memory.

On a small incident one person may hold several roles, but the roles are named
explicitly even then, so that handing one over is a sentence rather than a
negotiation.

## Communication cadence

The status page is updated within fifteen minutes of declaring a SEV1, and
every thirty minutes thereafter even when there is nothing new to report.
Silence is read by customers as absence of progress. Updates state what is
known, what is being done, and when the next update will arrive, and they
avoid speculating about cause while the investigation is open.

## Incident INC-4471: connection pool exhaustion

The quarterly reindex job opened twenty-four parallel workers against a pool
configured for ten connections. Requests queued behind the pool and eventually
timed out, producing a wave of gateway errors that looked like a database
outage. The database itself was healthy throughout.

Resolution was to cancel the reindex, after which the queue drained in under a
minute. The permanent fix separated batch workloads onto their own pool so an
analytical job can no longer starve interactive traffic. The lesson recorded
in the postmortem was that pool saturation presents as a downstream failure
and is easily misattributed.

## Incident INC-5238: certificate expiry

An intermediate certificate expired on a Saturday. Health checks continued to
pass because they were configured to skip verification, so the first signal
was customer reports. The remediation was to renew the certificate and to make
the health check verify the full chain, so the monitoring reflects what a real
client experiences.

## Incident INC-6012: cache stampede

A popular cache entry expired during peak traffic and several thousand
requests recomputed it simultaneously, saturating the service that produces
it. Latency tripled for four minutes until the entry repopulated. The fix
introduced a short randomised extension to expiry times so that entries
created together do not expire together, plus a single-flight guard that lets
one request recompute while the rest wait for its result.

## Incident INC-6390: log volume outage

A debug-level logger was left enabled after an investigation and filled the
log ingestion pipeline, which began dropping records. The visible symptom was
missing telemetry rather than a service fault, and it went unnoticed for two
days because the alert on ingestion volume was configured to fire on a drop to
zero rather than on saturation.

## Rolling back a release

Undoing a bad deployment means redeploying the previously published image tag.
The registry retains the last five tags for each service, so any of them can
be restored without a rebuild. Database migrations are not reversed
automatically: every migration is required to be backward compatible with the
preceding release, which is what makes an application rollback safe on its
own.

If a migration must be undone, that is a separate and deliberate operation
performed after the application has been rolled back and traffic is stable.

## Deployment strategy

Releases roll out to one instance first and pause for ten minutes while error
rate and latency are compared against the unchanged instances. If either
degrades beyond the configured margin the rollout halts automatically and the
single instance is reverted. Only after the pause does the remainder proceed,
in batches of a quarter of the fleet.

Deployments are frozen from Friday afternoon until Monday morning except for
fixes to an active incident, because the cost of a bad release is dominated by
how quickly someone notices it.

## Feature flags

Anything with meaningful risk ships behind a flag that is off by default, so
the deployment and the exposure are separate events. A flag that has been
fully on for two release cycles is removed, since a permanent flag is an
untested code path that accumulates.

Flags are evaluated at the edge and their state is included in request logs,
because an incident where behaviour differs per user is unresolvable without
knowing which flags each request saw.

## Error codes

Code E-1004 indicates that an upstream dependency returned a malformed
response. Code E-1007 indicates that a request exceeded its deadline before
any upstream was contacted, which usually points at saturation rather than a
slow dependency. Code E-2011 is reserved for authentication failures where the
token was well formed but its signature did not verify. Code E-3300 marks a
request rejected because the tenant exceeded its configured quota.

## Latency budgets

Budgets are stated at the ninety-ninth percentile and evaluated over a rolling
one-hour window. A request slower than eight hundred milliseconds counts
against the budget. Exhausting the budget freezes feature deployment for that
service until the following review, though emergency fixes are always allowed.

## Why percentiles rather than averages

An average conceals the shape of the distribution. A service where most
requests take twenty milliseconds and one in a hundred takes four seconds has
a healthy average and a population of users who consider it broken. Percentile
targets describe the experience of the unluckiest users, which is the group
that decides whether the product feels reliable.

Percentiles do not average across services either. Combining the ninety-ninth
percentile of two dependencies does not give the ninety-ninth percentile of a
request that calls both, and reasoning as though it does understates tail
latency badly.

## Retries and their hazards

A retry is appropriate only for an operation that is safe to perform twice and
for a failure that is plausibly transient. Retrying a request that timed out
because the service is saturated adds load to a system that is failing from
load, which is how a brief degradation becomes an outage.

Every retry uses exponential backoff with jitter, and every retry budget is
bounded so that a caller cannot amplify one upstream failure into many. A
circuit breaker stops retrying entirely once the failure rate crosses a
threshold, and probes occasionally to discover recovery.

## Timeouts

Every outbound call has an explicit timeout shorter than the deadline of the
request that triggered it. A call with no timeout inherits the operating
system default, which is measured in minutes and holds a connection long after
the client has given up. Nested calls must have decreasing budgets, so an
inner call cannot outlive the outer one that is waiting on it.

## Backups and their verification

Backups run nightly with a retention of thirty-five days, and a restore is
exercised into a scratch environment every month. A backup that has never been
restored is a hypothesis rather than a safeguard, and the failure mode is
discovered at the worst possible moment.

The restore drill measures elapsed time as well as success, because a recovery
objective stated in hours is meaningless if nobody has established how long a
restore of the current data volume actually takes.

## Access and credentials

Production access is granted for a limited window and expires automatically.
Standing administrative access is not issued to individuals. Credentials used
by services are rotated quarterly and on any suspicion of exposure, and no
credential is shared between environments so that a compromise in staging
cannot reach production.

## Capacity review

Capacity is reviewed quarterly against the previous period's peak rather than
its average, because the system must survive the worst hour rather than a
typical one. Forecasts assume traffic grows at the observed rate and that no
single dependency can absorb more than half of any planned increase.

## Load shedding

When saturation is unavoidable the system sheds load deliberately rather than
degrading uniformly. Requests are classified by importance, and the least
important are rejected quickly with a clear error so their callers can back
off. Rejecting cheaply is essential: a rejection that still consumes the
contended resource sheds nothing.

## Postmortems

A postmortem is written for every SEV1 and SEV2, published within five working
days, and reviewed in an open forum. It records the timeline, the contributing
factors, and the actions taken, and it names systems rather than individuals.
An action item without an owner and a date is not an action item, and the
review explicitly checks whether previous ones were completed.

## On-call expectations

An engineer on call is expected to be reachable and able to reach a working
network connection within fifteen minutes. Shifts run for one week. Anyone
paged outside working hours takes compensating time off, and sustained paging
volume is treated as a defect in the system rather than a property of the
rotation.
